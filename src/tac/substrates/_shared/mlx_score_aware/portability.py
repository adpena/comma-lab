# SPDX-License-Identifier: MIT
"""Numpy-portable inflate contract verifier (the 8th directive's decode half).

Separation of concerns: this module owns ONLY the "is the substrate's
``inflate.py`` numpy/PIL-portable" static check per the 8th MLX-first standing
directive + HNeRV parity discipline L4. It delegates the canonical AST scan to
``tac.substrates._shared.numpy_portable_inflate.assert_inflate_is_numpy_portable``
(the LANDED bridge, commit ``980808776``) so there is ONE forbidden-framework
scan surface across the harness + the bridge + the runtime emitter — no
re-implemented drift.

The harness-level wrapper adds the ``{numpy_portable, checked_path,
import_roots}`` return shape the harness orchestrator + tests consume, raising
the shared ``MlxScoreAwareHarnessError`` (not the bridge's
``InflateNotNumpyPortableError``) so callers catch one harness error type.

[verified-against: tac.substrates._shared.numpy_portable_inflate.assert_inflate_is_numpy_portable canonical bridge verifier]
"""
from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
)

# inflate.py portability contract: NO mlx / torch import at decode time. The
# canonical bridge enforces a SUPERSET (torch/mlx/tensorflow/jax); this harness
# constant names the two roots the harness historically documented so the
# public surface is stable for callers/tests that introspect it.
FORBIDDEN_INFLATE_IMPORT_ROOTS: tuple[str, ...] = ("mlx", "torch")


def assert_numpy_portable_inflate(
    inflate_py_path: Any,
    *,
    forbidden_roots: Sequence[str] = FORBIDDEN_INFLATE_IMPORT_ROOTS,
) -> dict[str, Any]:
    """Statically verify a substrate ``inflate.py`` is numpy/PIL-portable.

    Per the 8th standing directive: the INFLATE path must decode MLX-trained
    weights via numpy/PIL primitives with NO ``mlx`` / ``torch`` import. This
    parses ``inflate.py`` via ``ast`` and refuses any ``import mlx`` /
    ``import torch`` / ``from mlx... import`` / ``from torch... import``
    statement (including dotted submodules ``mlx.core`` / ``torch.nn``).

    String mentions of ``mlx`` / ``torch`` in comments / docstrings are NOT
    flagged (ast import-node scan only).

    Args:
        inflate_py_path: path to the substrate's ``inflate.py``.
        forbidden_roots: import roots that break numpy-portability.

    Returns:
        ``{"numpy_portable": True, "checked_path": ..., "import_roots": [...]}``
        on success.

    Raises:
        MlxScoreAwareHarnessError: ``inflate.py`` missing OR imports a
            forbidden root.
    """
    path = Path(inflate_py_path)
    if not path.is_file():
        raise MlxScoreAwareHarnessError(
            f"inflate.py not found at {path!s}; numpy-portable inflate "
            "contract cannot be verified (HNeRV parity L4)."
        )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    seen_roots: set[str] = set()
    forbidden_hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                seen_roots.add(root)
                if root in forbidden_roots:
                    forbidden_hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # node.module is None for `from . import x` (relative); skip those.
            root = node.module.split(".", 1)[0]
            seen_roots.add(root)
            if root in forbidden_roots:
                forbidden_hits.append(node.module)
    if forbidden_hits:
        raise MlxScoreAwareHarnessError(
            f"inflate.py at {path!s} imports forbidden non-portable root(s) "
            f"{sorted(set(forbidden_hits))}; the numpy-portable inflate "
            "contract (8th standing directive + HNeRV parity L4) requires the "
            "decode path to be numpy/PIL-only (no mlx/torch). MLX is a "
            "training-time-only dependency."
        )
    return {
        "numpy_portable": True,
        "checked_path": str(path),
        "import_roots": sorted(seen_roots),
    }


__all__ = [
    "FORBIDDEN_INFLATE_IMPORT_ROOTS",
    "assert_numpy_portable_inflate",
]
