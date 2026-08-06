"""Import helpers for the vendored PR130 pose-leg scripts.

PR130's scripts use flat imports such as ``from carrier_codec import ...`` and
``from semantic_renderer_oracle import ...``.  This module keeps that import
surface local and temporary so mx2 can run the pose scripts without editing the
borrowed files or the mx1 renderer lift.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

PACKAGE_DIR = Path(__file__).resolve().parent
PR130_LIFT_DIR = PACKAGE_DIR.parent
POSE_LIFTED_DIR = PACKAGE_DIR / "lifted"
MX1_RENDERER_LIFTED_DIR = PR130_LIFT_DIR / "lifted"


@contextmanager
def lifted_script_path() -> Iterator[None]:
    """Temporarily expose PR130 pose and mx1 renderer flat-import directories."""

    paths = [str(POSE_LIFTED_DIR), str(MX1_RENDERER_LIFTED_DIR)]
    previous = list(sys.path)
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = previous


def load_lifted_module(name: str) -> ModuleType:
    """Load a vendored PR130 pose module by filename without mutating it."""

    if "/" in name or name.endswith(".py"):
        raise ValueError("module name must be bare, for example 'carrier_codec'")
    path = POSE_LIFTED_DIR / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        f"tac.pr130_lift.pose.lifted_dynamic.{name}",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored PR130 module {name!r}")
    module = importlib.util.module_from_spec(spec)
    with lifted_script_path():
        spec.loader.exec_module(module)
    return module
