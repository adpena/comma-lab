from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_builder_module():
    module_path = REPO_ROOT / "tools" / "build_cross_paradigm_admm_x_op1_finalizer.py"
    spec = importlib.util.spec_from_file_location(
        "build_cross_paradigm_admm_x_op1_finalizer", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_output_root_accepts_repo_relative_paths() -> None:
    builder = _load_builder_module()

    resolved = builder._resolve_output_root(Path("experiments/results/worker_d"))

    assert resolved.is_absolute()
    assert resolved == (REPO_ROOT / "experiments/results/worker_d").resolve()


def test_resolve_output_root_preserves_absolute_paths(tmp_path: Path) -> None:
    builder = _load_builder_module()
    output_root = tmp_path / "worker-d-out"

    assert builder._resolve_output_root(output_root) == output_root.resolve()
