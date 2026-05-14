# SPDX-License-Identifier: MIT
"""F3 GTScorerCache wire-in coverage for priority substrate losses."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest


_PRIORITY_LOSS_CLASSES: tuple[tuple[str, str], ...] = (
    ("tac.substrates.c6_e4_mdl_ibps.score_aware_loss", "MDLIBPSScoreAwareLoss"),
    (
        "tac.substrates.d4_wyner_ziv_frame_0.score_aware_loss",
        "WynerZivFrame0ScoreAwareLoss",
    ),
    (
        "tac.substrates.z4_cooperative_receiver_loss.score_aware_loss",
        "CooperativeReceiverScoreAwareLoss",
    ),
    (
        "tac.substrates.z5_predictive_coding_world_model.score_aware_loss",
        "PredictiveCodingScoreAwareLoss",
    ),
    (
        "tac.substrates.sabor_boundary_only_renderer.score_aware_loss",
        "SaborBoundaryOnlyScoreAwareLoss",
    ),
    ("tac.substrates.s2sbs_byte_stuffing.score_aware_loss", "S2sbsScoreAwareLoss"),
    (
        "tac.substrates.a1_plus_wavelet_residual.score_aware_loss",
        "A1PlusWaveletResidualScoreAwareLoss",
    ),
    (
        "tac.substrates.a1_plus_lapose.score_aware_loss",
        "A1PlusLaposeScoreAwareLoss",
    ),
    ("tac.substrates.wavelet.score_aware_loss", "WaveletScoreAwareLoss"),
    ("tac.substrates.siren.score_aware_loss", "SirenScoreAwareLoss"),
)


@pytest.mark.parametrize(("module_name", "class_name"), _PRIORITY_LOSS_CLASSES)
def test_priority_loss_forward_accepts_canonical_cache_kwargs(
    module_name: str,
    class_name: str,
) -> None:
    cls = getattr(importlib.import_module(module_name), class_name)
    sig = inspect.signature(cls.forward)
    for name in ("gt_pose_batch", "gt_seg_batch", "gt_seg_already_probs"):
        assert name in sig.parameters, f"{class_name} missing {name}"
        assert sig.parameters[name].default is None


@pytest.mark.parametrize(("module_name", "class_name"), _PRIORITY_LOSS_CLASSES)
def test_priority_loss_routes_through_shared_dispatch_helper(
    module_name: str,
    class_name: str,
) -> None:
    cls = getattr(importlib.import_module(module_name), class_name)
    source = inspect.getsource(cls.forward)
    assert "score_pair_components_dispatch" in source


def test_z3_lagrangian_accepts_canonical_cache_kwargs_and_dispatches() -> None:
    from tac.substrates.z3_balle_hyperprior_bolton.score_aware_loss import (
        z3_lagrangian,
    )

    sig = inspect.signature(z3_lagrangian)
    for name in ("gt_pose_batch", "gt_seg_batch", "gt_seg_already_probs"):
        assert name in sig.parameters
        assert sig.parameters[name].default is None
    assert "score_pair_components_dispatch" in inspect.getsource(z3_lagrangian)


@pytest.mark.parametrize(
    "trainer_path",
    (
        "experiments/train_substrate_siren.py",
        "experiments/train_substrate_wavelet.py",
        "experiments/train_substrate_sabor_boundary_only_renderer.py",
        "experiments/train_substrate_c6_e4_mdl_ibps.py",
        "experiments/train_substrate_d4_wyner_ziv_frame_0.py",
        "experiments/train_substrate_a1_plus_wavelet_residual.py",
        "experiments/train_substrate_a1_plus_lapose.py",
    ),
)
def test_indexed_priority_trainers_build_and_lookup_gt_cache(trainer_path: str) -> None:
    text = Path(trainer_path).read_text(encoding="utf-8")
    assert "build_optimized_training_context" in text
    assert "gt_cache.lookup(" in text
    assert "gt_pose_batch=" in text
    assert "--disable-gt-scorer-cache" in text


@pytest.mark.parametrize(
    "trainer_path",
    (
        "experiments/train_substrate_c6_e4_mdl_ibps.py",
        "experiments/train_substrate_d4_wyner_ziv_frame_0.py",
        "experiments/train_substrate_a1_plus_wavelet_residual.py",
        "experiments/train_substrate_a1_plus_lapose.py",
    ),
)
def test_full_priority_trainers_emit_proxy_axis_labels(trainer_path: str) -> None:
    text = Path(trainer_path).read_text(encoding="utf-8")
    assert "trainer_proxy_axis" in text
    assert "trainer_proxy_promotion_requirement" in text
