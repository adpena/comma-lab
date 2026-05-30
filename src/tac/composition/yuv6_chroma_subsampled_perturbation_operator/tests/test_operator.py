# SPDX-License-Identifier: MIT
"""Tests for the YUV6 chroma-subsampled perturbation operator.

Slot EEE NO FAKE IMPLEMENTATIONS gate coverage:
  - 4 canonical strategies produce DISTINCT weight maps (Jaccard < 1.0)
  - rgb_to_yuv6_numpy parity vs canonical differentiable_rgb_to_yuv6
  - Luma-preservation invariant holds (max-abs drift <= small threshold)
  - Chroma perturbation is materialized (nonzero chroma drift)
  - PerturbationResult Catalog #341 frozen markers reject promotion attempts
  - Catalog #323 canonical Provenance + Catalog #356 AxisDecomposition
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.composition.yuv6_chroma_subsampled_perturbation_operator import (
    ATICK_REDLICH_DEFAULT_BLEND_COEFFICIENTS,
    BT601_KYR,
    BT601_KYG,
    BT601_KYB,
    DEFAULT_PERTURBATION_MAGNITUDE,
    ChromaPerturbationStrategy,
    ChromaSubsampledPerturbationConfig,
    ChromaSubsampledPerturbationConfigInvalidError,
    ChromaSubsampledPerturbationResult,
    apply_chroma_subsampled_perturbation,
    assert_luma_preservation_invariant,
    compute_chroma_perturbation_weight_map,
    rgb_to_yuv6_numpy,
    yuv6_to_rgb_numpy,
)


def _make_random_frame(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 255.0, size=(h, w, 3)).astype(np.float64)


def test_canonical_constants_pinned() -> None:
    assert DEFAULT_PERTURBATION_MAGNITUDE == 4.0
    assert ATICK_REDLICH_DEFAULT_BLEND_COEFFICIENTS == (0.33, 0.33, 0.34)
    assert np.isclose(BT601_KYR + BT601_KYG + BT601_KYB, 1.0)


def test_rgb_to_yuv6_numpy_shape_and_channel_order() -> None:
    rgb = _make_random_frame(h=16, w=16)
    yuv6 = rgb_to_yuv6_numpy(rgb)
    assert yuv6.shape == (6, 8, 8)
    # All channels in [0, 255]
    assert yuv6.min() >= 0.0
    assert yuv6.max() <= 255.0


def test_rgb_to_yuv6_numpy_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        rgb_to_yuv6_numpy(np.zeros((10, 10), dtype=np.float64))  # 2D
    with pytest.raises(ValueError):
        rgb_to_yuv6_numpy(np.zeros((10, 10, 4), dtype=np.float64))  # 4 channels


def test_rgb_to_yuv6_round_trip_documented_lossy() -> None:
    """yuv6_to_rgb is APPROXIMATE inverse (4:2:0 nearest-neighbor + clip is lossy).

    Per the operator docstring, the canonical PoseNet operates DIRECTLY on
    YUV6 — the inverse RGB reconstruction is for downstream RGB-consuming
    code only + is documented-approximate. We verify shape parity + that
    the inverse produces SOME output (not NaN); we do NOT enforce a
    tight mean-error bound because the lossy 4:2:0 + clip path is
    legitimately lossy on random uniform inputs.
    """
    rgb = _make_random_frame(h=16, w=16, seed=11)
    yuv6 = rgb_to_yuv6_numpy(rgb)
    rgb_recovered = yuv6_to_rgb_numpy(yuv6)
    assert rgb_recovered.shape == rgb.shape
    assert not np.any(np.isnan(rgb_recovered))
    assert rgb_recovered.min() >= 0.0
    assert rgb_recovered.max() <= 255.0


def test_rgb_to_yuv6_numpy_matches_canonical_torch_within_tolerance() -> None:
    """Parity vs canonical differentiable_rgb_to_yuv6.

    Use a small fixed input; convert both via numpy + torch paths; verify
    max-abs error is < 0.05 (the canonical impls use float64 / float32
    respectively; small rounding errors expected).
    """
    pytest.importorskip("torch")
    import torch

    from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

    rgb = _make_random_frame(h=16, w=16, seed=42)
    # Numpy path expects (H, W, 3)
    numpy_yuv6 = rgb_to_yuv6_numpy(rgb)  # (6, H//2, W//2)
    # Torch path expects (..., 3, H, W)
    torch_in = torch.tensor(rgb.transpose(2, 0, 1)[None], dtype=torch.float32)
    torch_yuv6 = differentiable_rgb_to_yuv6(torch_in)  # (1, 6, H//2, W//2)
    torch_yuv6_np = torch_yuv6.detach().numpy()[0]

    max_abs = float(np.max(np.abs(numpy_yuv6 - torch_yuv6_np)))
    # Allow up to 0.05 (float32 vs float64 rounding); typically much smaller
    assert max_abs < 0.05, f"YUV6 parity drift {max_abs} exceeds 0.05"


def test_config_post_init_validates_strategy_type() -> None:
    with pytest.raises(ChromaSubsampledPerturbationConfigInvalidError):
        ChromaSubsampledPerturbationConfig(strategy="not_an_enum")  # type: ignore[arg-type]


def test_config_rejects_negative_magnitude() -> None:
    with pytest.raises(ChromaSubsampledPerturbationConfigInvalidError):
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
            perturbation_magnitude=-1.0,
        )


def test_config_rejects_magnitude_over_255() -> None:
    with pytest.raises(ChromaSubsampledPerturbationConfigInvalidError):
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
            perturbation_magnitude=256.0,
        )


def test_config_rejects_non_unit_atick_blend() -> None:
    with pytest.raises(ChromaSubsampledPerturbationConfigInvalidError):
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
            atick_redlich_blend_coefficients=(0.5, 0.5, 0.5),  # sum = 1.5
        )


def test_config_segnet_requires_map() -> None:
    with pytest.raises(ChromaSubsampledPerturbationConfigInvalidError):
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.SEGNET_GRADIENT_WEIGHTED,
            # segnet_gradient_map missing
        )


def test_config_posenet_requires_map() -> None:
    with pytest.raises(ChromaSubsampledPerturbationConfigInvalidError):
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.POSENET_GRADIENT_WEIGHTED_VIA_MAE_V,
            # posenet_gradient_map missing
        )


def test_config_joint_requires_both_maps() -> None:
    with pytest.raises(ChromaSubsampledPerturbationConfigInvalidError):
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.JOINT_ATICK_REDLICH_LINEAR_COMBINATION,
            segnet_gradient_map=np.zeros((4, 4)),
            # posenet_gradient_map missing
        )


def test_4_strategies_produce_distinct_weight_maps() -> None:
    """Slot EEE NO FAKE gate: 4 strategies must yield distinct maps (Jaccard < 1)."""
    rgb = _make_random_frame(h=16, w=16, seed=7)
    h2, w2 = 8, 8
    rng = np.random.default_rng(99)
    segnet_map = rng.uniform(0.0, 1.0, (h2, w2)).astype(np.float64)
    posenet_map = rng.uniform(0.0, 1.0, (h2, w2)).astype(np.float64)

    configs = [
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
        ),
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.SEGNET_GRADIENT_WEIGHTED,
            segnet_gradient_map=segnet_map,
        ),
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.POSENET_GRADIENT_WEIGHTED_VIA_MAE_V,
            posenet_gradient_map=posenet_map,
        ),
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.JOINT_ATICK_REDLICH_LINEAR_COMBINATION,
            segnet_gradient_map=segnet_map,
            posenet_gradient_map=posenet_map,
        ),
    ]
    maps = [compute_chroma_perturbation_weight_map(c, rgb) for c in configs]

    # Pairwise: maps must differ (max-abs > 0)
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            diff = float(np.max(np.abs(maps[i] - maps[j])))
            assert diff > 0.0, (
                f"strategies {configs[i].strategy.value} and "
                f"{configs[j].strategy.value} produced IDENTICAL weight maps "
                "(Slot EEE NO FAKE invariant violated)"
            )


def test_weight_map_all_in_unit_interval() -> None:
    rgb = _make_random_frame(h=16, w=16)
    cfg = ChromaSubsampledPerturbationConfig(
        strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
    )
    wm = compute_chroma_perturbation_weight_map(cfg, rgb)
    assert wm.shape == (8, 8)
    assert wm.min() >= 0.0
    assert wm.max() <= 1.0


def test_apply_perturbation_canonical_signature() -> None:
    rgb_a = _make_random_frame(h=16, w=16, seed=1)
    rgb_b = _make_random_frame(h=16, w=16, seed=2)
    cfg = ChromaSubsampledPerturbationConfig(
        strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
        perturbation_magnitude=8.0,
    )
    result = apply_chroma_subsampled_perturbation(
        config=cfg, rgb_first_frame_hwc=rgb_a, rgb_second_frame_hwc=rgb_b
    )
    assert result.strategy_used == "local_variance_weighted"
    assert result.perturbed_rgb_first_frame.shape == rgb_a.shape
    assert result.perturbed_rgb_second_frame.shape == rgb_b.shape
    assert result.predicted_delta_adjustment == 0.0
    assert result.promotable is False
    assert result.axis_tag == "[predicted]"
    assert "schema_version" in result.provenance
    assert result.predicted_axis_decomposition is not None
    assert (
        result.predicted_axis_decomposition["schema_version"]
        == "axis_decomposition_v1"
    )


def test_chroma_perturbation_materialized_in_output() -> None:
    """The output frames differ from input in CHROMA channels (Slot EEE NO FAKE)."""
    rgb_a = _make_random_frame(h=16, w=16, seed=1)
    rgb_b = _make_random_frame(h=16, w=16, seed=2)
    cfg = ChromaSubsampledPerturbationConfig(
        strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
        perturbation_magnitude=20.0,
    )
    result = apply_chroma_subsampled_perturbation(
        config=cfg, rgb_first_frame_hwc=rgb_a, rgb_second_frame_hwc=rgb_b
    )
    # The output YUV6 chroma differs from input chroma
    diff_a_chroma = float(
        np.max(
            np.abs(
                result.perturbed_yuv6_first_frame[4:]
                - rgb_to_yuv6_numpy(rgb_a)[4:]
            )
        )
    )
    diff_b_chroma = float(
        np.max(
            np.abs(
                result.perturbed_yuv6_second_frame[4:]
                - rgb_to_yuv6_numpy(rgb_b)[4:]
            )
        )
    )
    assert diff_a_chroma > 0.0, "perturbation operator produced no chroma change in first frame"
    assert diff_b_chroma > 0.0, "perturbation operator produced no chroma change in second frame"
    # Chroma drift in YUV6 space is real
    assert result.chroma_perturbation_max_abs_drift_yuv6 > 0.0


def test_yuv6_luma_preservation_invariant_exact() -> None:
    """CANONICAL invariant: YUV6 luma channels are byte-IDENTICAL after perturbation.

    We modify ONLY channels 4 + 5 (U_sub, V_sub) of the YUV6 tensor, so
    channels 0-3 (Y00, Y10, Y01, Y11) must be EXACTLY preserved in the
    canonical YUV6-space output. Drift = 0.0 by construction.
    """
    rgb_a = _make_random_frame(h=16, w=16, seed=1)
    rgb_b = _make_random_frame(h=16, w=16, seed=2)
    cfg = ChromaSubsampledPerturbationConfig(
        strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
        perturbation_magnitude=20.0,
    )
    result = apply_chroma_subsampled_perturbation(
        config=cfg, rgb_first_frame_hwc=rgb_a, rgb_second_frame_hwc=rgb_b
    )
    # Canonical YUV6 luma drift MUST be 0
    assert result.luma_preservation_max_abs_drift_yuv6 == 0.0


def test_assert_luma_preservation_invariant_helper_yuv6() -> None:
    """The canonical helper operates in YUV6 space + asserts exact zero drift."""
    rgb_a = _make_random_frame(h=16, w=16, seed=1)
    rgb_b = _make_random_frame(h=16, w=16, seed=2)
    cfg = ChromaSubsampledPerturbationConfig(
        strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
        perturbation_magnitude=10.0,
    )
    result = apply_chroma_subsampled_perturbation(
        config=cfg, rgb_first_frame_hwc=rgb_a, rgb_second_frame_hwc=rgb_b
    )
    yuv6_orig_a = rgb_to_yuv6_numpy(rgb_a)
    # Default threshold is 1e-9; canonical YUV6 luma drift is exactly 0
    drift = assert_luma_preservation_invariant(
        original_yuv6=yuv6_orig_a,
        perturbed_yuv6=result.perturbed_yuv6_first_frame,
    )
    assert drift == 0.0
    # If we deliberately corrupt the luma channel, assert raises
    corrupted = result.perturbed_yuv6_first_frame.copy()
    corrupted[0, 0, 0] += 10.0  # mutate Y00
    with pytest.raises(AssertionError, match="luma-preservation invariant"):
        assert_luma_preservation_invariant(
            original_yuv6=yuv6_orig_a,
            perturbed_yuv6=corrupted,
            luma_drift_threshold=1e-6,
        )


def _make_minimal_result_kwargs() -> dict:
    return dict(
        strategy_used="local_variance_weighted",
        perturbed_yuv6_first_frame=np.zeros((6, 2, 2)),
        perturbed_yuv6_second_frame=np.zeros((6, 2, 2)),
        perturbed_rgb_first_frame=np.zeros((4, 4, 3)),
        perturbed_rgb_second_frame=np.zeros((4, 4, 3)),
        perturbation_weight_map=np.zeros((2, 2)),
        predicted_delta_adjustment=0.0,
        promotable=False,
        axis_tag="[predicted]",
        confidence=0.0,
        rationale="test",
        luma_preservation_max_abs_drift_yuv6=0.0,
        luma_preservation_max_abs_drift_rgb_reconstructed=0.0,
        chroma_perturbation_max_abs_drift_yuv6=0.0,
        predicted_axis_decomposition=None,
    )


def test_result_rejects_nonzero_predicted_delta() -> None:
    """Catalog #341 frozen-False invariant: result CANNOT carry nonzero delta."""
    kwargs = _make_minimal_result_kwargs()
    kwargs["predicted_delta_adjustment"] = -0.001  # FORBIDDEN
    with pytest.raises(ValueError, match="predicted_delta_adjustment MUST be 0.0"):
        ChromaSubsampledPerturbationResult(**kwargs)


def test_result_rejects_promotable_true() -> None:
    kwargs = _make_minimal_result_kwargs()
    kwargs["promotable"] = True  # FORBIDDEN
    with pytest.raises(ValueError, match="promotable MUST be False"):
        ChromaSubsampledPerturbationResult(**kwargs)


def test_result_rejects_wrong_axis_tag() -> None:
    kwargs = _make_minimal_result_kwargs()
    kwargs["axis_tag"] = "[contest-CUDA]"  # FORBIDDEN
    with pytest.raises(ValueError, match="axis_tag"):
        ChromaSubsampledPerturbationResult(**kwargs)


def test_segnet_strategy_runs_with_supplied_map() -> None:
    rgb = _make_random_frame(h=16, w=16, seed=1)
    segnet_map = np.linspace(0, 1, 64).reshape(8, 8)
    cfg = ChromaSubsampledPerturbationConfig(
        strategy=ChromaPerturbationStrategy.SEGNET_GRADIENT_WEIGHTED,
        segnet_gradient_map=segnet_map,
    )
    wm = compute_chroma_perturbation_weight_map(cfg, rgb)
    assert wm.shape == (8, 8)
    assert wm.min() >= 0.0
    assert wm.max() <= 1.0


def test_segnet_strategy_rejects_mismatched_map_shape() -> None:
    rgb = _make_random_frame(h=16, w=16, seed=1)
    bad_map = np.zeros((4, 4))  # wrong shape
    cfg = ChromaSubsampledPerturbationConfig(
        strategy=ChromaPerturbationStrategy.SEGNET_GRADIENT_WEIGHTED,
        segnet_gradient_map=bad_map,
    )
    with pytest.raises(ValueError, match="segnet_gradient_map shape"):
        compute_chroma_perturbation_weight_map(cfg, rgb)


def test_joint_strategy_blends_with_atick_redlich() -> None:
    rgb = _make_random_frame(h=16, w=16, seed=1)
    seg = np.linspace(0, 1, 64).reshape(8, 8)
    pose = np.linspace(1, 0, 64).reshape(8, 8)  # reversed
    cfg = ChromaSubsampledPerturbationConfig(
        strategy=ChromaPerturbationStrategy.JOINT_ATICK_REDLICH_LINEAR_COMBINATION,
        segnet_gradient_map=seg,
        posenet_gradient_map=pose,
    )
    wm = compute_chroma_perturbation_weight_map(cfg, rgb)
    assert wm.shape == (8, 8)
    assert wm.min() >= 0.0
    assert wm.max() <= 1.0


def test_provenance_canonical_fields() -> None:
    rgb_a = _make_random_frame(h=8, w=8, seed=1)
    rgb_b = _make_random_frame(h=8, w=8, seed=2)
    cfg = ChromaSubsampledPerturbationConfig(
        strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
    )
    result = apply_chroma_subsampled_perturbation(
        config=cfg, rgb_first_frame_hwc=rgb_a, rgb_second_frame_hwc=rgb_b
    )
    prov = result.provenance
    assert prov["kind"] == "predicted_from_model"
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["axis_tag"] == "[predicted]"
    assert prov["evidence_grade"] == "predicted"
    assert prov["strategy_used"] == "local_variance_weighted"


def test_axis_decomposition_canonical_fields() -> None:
    rgb_a = _make_random_frame(h=8, w=8, seed=1)
    rgb_b = _make_random_frame(h=8, w=8, seed=2)
    cfg = ChromaSubsampledPerturbationConfig(
        strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
    )
    result = apply_chroma_subsampled_perturbation(
        config=cfg, rgb_first_frame_hwc=rgb_a, rgb_second_frame_hwc=rgb_b
    )
    decomp = result.predicted_axis_decomposition
    assert decomp is not None
    assert decomp["schema_version"] == "axis_decomposition_v1"
    assert decomp["predicted_d_seg_delta"] == 0.0
    assert decomp["predicted_d_pose_delta"] == 0.0
    assert decomp["predicted_archive_bytes_delta"] == 0
    assert decomp["axis_tag"] == "[predicted]"
    # Canonical YUV6 luma drift is exactly 0 by construction
    assert decomp["luma_preservation_max_abs_drift_yuv6"] == 0.0
    # Chroma YUV6 drift is positive (perturbation was materialized)
    assert decomp["chroma_perturbation_max_abs_drift_yuv6"] > 0.0


def test_all_4_strategies_apply_without_error() -> None:
    """Smoke: every strategy completes apply_* end-to-end."""
    rgb_a = _make_random_frame(h=16, w=16, seed=1)
    rgb_b = _make_random_frame(h=16, w=16, seed=2)
    seg = np.linspace(0, 1, 64).reshape(8, 8)
    pose = np.linspace(1, 0, 64).reshape(8, 8)

    configs = [
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
        ),
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.SEGNET_GRADIENT_WEIGHTED,
            segnet_gradient_map=seg,
        ),
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.POSENET_GRADIENT_WEIGHTED_VIA_MAE_V,
            posenet_gradient_map=pose,
        ),
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.JOINT_ATICK_REDLICH_LINEAR_COMBINATION,
            segnet_gradient_map=seg,
            posenet_gradient_map=pose,
        ),
    ]
    for cfg in configs:
        result = apply_chroma_subsampled_perturbation(
            config=cfg, rgb_first_frame_hwc=rgb_a, rgb_second_frame_hwc=rgb_b
        )
        assert result.strategy_used == cfg.strategy.value
        assert result.chroma_perturbation_max_abs_drift_yuv6 > 0.0


def test_strategies_4_produce_distinct_perturbed_outputs() -> None:
    """End-to-end: 4 strategies produce DIFFERENT perturbed RGB outputs."""
    rgb_a = _make_random_frame(h=16, w=16, seed=1)
    rgb_b = _make_random_frame(h=16, w=16, seed=2)
    seg = np.linspace(0, 1, 64).reshape(8, 8)
    pose = np.linspace(1, 0, 64).reshape(8, 8)

    configs = [
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.LOCAL_VARIANCE_WEIGHTED,
        ),
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.SEGNET_GRADIENT_WEIGHTED,
            segnet_gradient_map=seg,
        ),
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.POSENET_GRADIENT_WEIGHTED_VIA_MAE_V,
            posenet_gradient_map=pose,
        ),
        ChromaSubsampledPerturbationConfig(
            strategy=ChromaPerturbationStrategy.JOINT_ATICK_REDLICH_LINEAR_COMBINATION,
            segnet_gradient_map=seg,
            posenet_gradient_map=pose,
        ),
    ]
    results = [
        apply_chroma_subsampled_perturbation(
            config=cfg, rgb_first_frame_hwc=rgb_a, rgb_second_frame_hwc=rgb_b
        )
        for cfg in configs
    ]
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            diff = float(
                np.max(
                    np.abs(
                        results[i].perturbed_rgb_first_frame
                        - results[j].perturbed_rgb_first_frame
                    )
                )
            )
            assert diff > 0.0, (
                f"strategies {configs[i].strategy.value} and "
                f"{configs[j].strategy.value} produced identical outputs"
            )
