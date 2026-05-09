from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments/modal_phase_a1_score_gradient_pr101.py"


def _load_modal_phase_a1_module():
    name = "_modal_phase_a1_recover_paths_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_modal_phase_a1_installs_local_and_remote_repo_import_paths() -> None:
    module = _load_modal_phase_a1_module()

    expected = {
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "upstream"),
        str(REPO_ROOT),
        str(module.REMOTE_REPO / "src"),
        str(module.REMOTE_REPO / "upstream"),
        str(module.REMOTE_REPO),
    }
    assert expected.issubset(set(sys.path))
