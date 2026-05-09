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


def test_modal_phase_a1_normalizes_canonical_auth_eval_schema() -> None:
    module = _load_modal_phase_a1_module()

    metrics = module._eval_metric_summary(
        {
            "canonical_score": 0.22655968711150934,
            "score_recomputed_from_components": 0.22655968711150934,
            "final_score": 0.23,
            "avg_posenet_dist": 0.00017099,
            "avg_segnet_dist": 0.000665,
            "score_rate_contribution": 0.11870875,
            "rate_unscaled": 0.00474835,
        }
    )

    assert metrics == {
        "score": 0.22655968711150934,
        "pose_avg": 0.00017099,
        "seg_avg": 0.000665,
        "rate": 0.11870875,
        "rate_unscaled": 0.00474835,
    }


def test_modal_phase_a1_normalizes_legacy_score_components_schema() -> None:
    module = _load_modal_phase_a1_module()

    metrics = module._eval_metric_summary(
        {
            "score": "0.31",
            "score_components": {
                "pose": "0.0002",
                "seg": 0.0006,
                "rate": 0.119,
                "rate_unscaled": 0.00476,
            },
        }
    )

    assert metrics == {
        "score": 0.31,
        "pose_avg": 0.0002,
        "seg_avg": 0.0006,
        "rate": 0.119,
        "rate_unscaled": 0.00476,
    }
