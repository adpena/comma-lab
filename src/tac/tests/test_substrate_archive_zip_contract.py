# SPDX-License-Identifier: MIT
"""Regression tests for substrate trainer charged archive.zip boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_ONLY_TRAINERS = [
    "experiments/train_substrate_siren.py",
    "experiments/train_substrate_sabor_boundary_only_renderer.py",
    "experiments/train_substrate_balle_renderer.py",
    "experiments/train_substrate_cool_chic.py",
    "experiments/train_substrate_vq_vae.py",
    "experiments/train_substrate_wavelet.py",
]


def _build_archive_zip_function(path: Path) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_build_archive_zip":
            return node
    raise AssertionError(f"{path} does not define _build_archive_zip")


def _string_constants(node: ast.AST) -> set[str]:
    return {child.value for child in ast.walk(node) if isinstance(child, ast.Constant) and isinstance(child.value, str)}


def test_substrate_trainers_build_data_only_charged_archives() -> None:
    """archive.zip must contain data bytes only; runtime files are external custody."""

    for rel in DATA_ONLY_TRAINERS:
        func = _build_archive_zip_function(REPO_ROOT / rel)
        constants = _string_constants(func)
        assert "0.bin" in constants, rel
        assert "inflate.sh" not in constants, rel
        assert "inflate.py" not in constants, rel
        assert "src" not in constants, rel
