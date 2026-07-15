# SPDX-License-Identifier: MIT
"""Adversarial tests for the task-495 pose-blind compute gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.witness_dsl import curriculum_dsl as cd
from tac.witness_dsl import lever_registry

REPO = Path(__file__).resolve().parents[3]
TRAINER = REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"


def _load_trainer():
    module_name = "task494_pose_gate_levelset_trainer"
    if module_name in sys.modules:
        return sys.modules[module_name]
    for path in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(module_name, TRAINER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_pose_verdict_gate_names_fail_closed_at_composition() -> None:
    names = lever_registry.name_composable_levers()
    assert "PoseVerdictGate" not in names
    assert "PoseVerdictGateDryStart" not in names
    assert "PoseBlindComputeGate" in names
    lever = lever_registry.resolve_composable_lever("PoseBlindComputeGate")
    assert lever.overrides == {
        "--pose-training-compute-gate": True,
        "--verdict-pose-gate": True,
    }


def test_pose_verdict_gate_rejects_every_legacy_value() -> None:
    with pytest.raises(ValueError, match="no payload-bound pose cache"):
        cd.PoseVerdictGate(canary_every=1, banked_r1_dpose=0.0)


def test_real_parser_preserves_only_disabled_legacy_parse_surface() -> None:
    parser = cd.build_real_trainer_parser()
    defaults = parser.parse_args(["--out-dir", "task494_parser_only"])
    assert defaults.verdict_pose_gate is False
    assert defaults.pose_training_compute_gate is False
    assert defaults.verdict_pose_canary_every == 8
    assert not hasattr(defaults, "banked_r1_dpose")
    armed = parser.parse_args(
        [
            "--out-dir",
            "task494_parser_only",
            "--verdict-pose-gate",
            "--pose-training-compute-gate",
            "--verdict-pose-canary-every",
            "3",
        ]
    )
    assert armed.verdict_pose_gate is True
    assert armed.pose_training_compute_gate is True
    assert armed.verdict_pose_canary_every == 3


def test_cpu_chunked_skip_never_calls_posenet(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_trainer()
    pose_calls = 0

    def fake_seg(_model, frames, _labels):
        return [0.25] * len(frames)

    def forbidden_pose(*_args, **_kwargs):
        nonlocal pose_calls
        pose_calls += 1
        raise AssertionError("PoseNet must not run on a gated verdict")

    monkeypatch.setattr(module, "cpu_verdict_d_seg_batch", fake_seg)
    monkeypatch.setattr(module, "cpu_verdict_d_pose_batch", forbidden_pose)
    d_seg, d_pose = module._verdict_dseg_dpose_chunked(
        object(),
        object(),
        [object(), object()],
        [object(), object()],
        [object(), object()],
        [object(), object()],
        vbatch=1,
        compute_pose=False,
    )
    assert d_seg == pytest.approx(0.25)
    assert d_pose is None
    assert pose_calls == 0


def test_gpu_seg_only_skip_never_calls_gpu_posenet(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_trainer()
    seg_calls = 0

    def fake_gpu_seg(_mlx_segnet, _seg_cpu, frames, _labels):
        nonlocal seg_calls
        seg_calls += 1
        realized = np.zeros((len(frames), 2, 3), dtype=np.int64)
        return [0.125] * len(frames), realized

    monkeypatch.setattr(module, "gpu_verdict_d_seg_argmax_batch", fake_gpu_seg)
    d_seg, realized = module._gpu_verdict_dseg_chunked(
        SimpleNamespace(segnet=object()),
        object(),
        [object(), object(), object()],
        [object(), object(), object()],
        vbatch=2,
        return_realized=True,
    )
    assert d_seg == pytest.approx(0.125)
    assert len(realized) == 3
    assert seg_calls == 2


def test_wirein_covers_all_branches_and_resume_counter() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    for required in (
        "compute_pose=_pose_decision.compute_live",
        "_verdict_subprocess_on and _pose_decision.compute_live",
        "_gpu_verdict_dseg_chunked(",
        "_verdict_dseg_dpose_nucleus_chunked(",
        "check_pose_verdict_fallback_is_live_or_refused(",
        "compute_pose=_compute_pose",
        "_optional_implied_score(",
        '_pose_gate_v0 = v0.pop("_pose_gate_telemetry", None)',
    ):
        assert required in source
    for forbidden in ("banked_pose_telemetry(",):
        assert forbidden not in source
