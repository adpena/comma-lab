"""Shared test helpers for importing repository-local tool scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_python_module(path: str | Path, module_name: str) -> ModuleType:
    """Import one Python file by path under a deterministic module name."""

    target = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_repo_tool(repo_root: str | Path, relative_path: str | Path, module_name: str) -> ModuleType:
    """Import a repository tool script by path relative to ``repo_root``."""

    return load_python_module(Path(repo_root) / relative_path, module_name)
