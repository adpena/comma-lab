from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tac.deploy.modal import runtime as modal_runtime


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


def test_modal_phase_a1_uses_shared_modal_runtime_for_scorer_deps(monkeypatch) -> None:
    module = _load_modal_phase_a1_module()
    calls: list[dict[str, object]] = []

    class FakeImage:
        def __init__(self) -> None:
            self.commands: tuple[str, ...] = ()

        def run_commands(self, *commands: str) -> "FakeImage":
            self.commands = commands
            return self

    fake_image = FakeImage()

    def fake_build_contest_cuda_base_image(modal_module, **kwargs):
        calls.append({"modal_module": modal_module, "kwargs": kwargs})
        return fake_image

    monkeypatch.setattr(
        module,
        "build_contest_cuda_base_image",
        fake_build_contest_cuda_base_image,
    )

    returned = module._build_phase_a1_base_image("fake-modal")

    assert returned is fake_image
    assert calls == [
        {"modal_module": "fake-modal", "kwargs": {"python_version": "3.11"}}
    ]
    assert "extra_pip_packages" not in calls[0]["kwargs"]
    assert fake_image.commands == module.FFMPEG_MASTER_INSTALL_COMMANDS


def test_modal_phase_a1_scorer_import_probe_uses_shared_runtime_modules() -> None:
    module = _load_modal_phase_a1_module()

    assert module.CONTEST_SCORER_IMPORT_PROBE_MODULES is (
        modal_runtime.CONTEST_SCORER_IMPORT_PROBE_MODULES
    )
    assert module.DALI_DISABLE_NVML_VALUE == modal_runtime.DALI_DISABLE_NVML_VALUE
    assert (
        module.PYTORCH_CUDA_ALLOC_CONF_VALUE
        == modal_runtime.PYTORCH_CUDA_ALLOC_CONF_VALUE
    )

    probe = module._scorer_runtime_import_probe_cmd()
    probe_source = probe[-1]
    for dependency in modal_runtime.CONTEST_SCORER_IMPORT_PROBE_MODULES:
        assert f"import {dependency}" in probe_source
    assert "from tac.scorer import load_differentiable_scorers" in probe_source


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
