from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "tools" / "run_full_scorer_vjp_from_scorer_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_full_scorer_vjp_from_scorer_cache", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vjp_array_stats_and_pair_l2_are_deterministic() -> None:
    module = _load_module()
    arr = np.array([[[[1.0, -2.0], [0.0, 3.0]]]], dtype=np.float32)
    stats = module._array_stats(arr)
    assert stats["abs_sum"] == 6.0
    assert stats["abs_mean"] == 1.5
    assert stats["abs_max"] == 3.0
    np.testing.assert_allclose(module._per_pair_l2(arr), np.array([np.sqrt(14.0)], dtype=np.float32))


def test_full_d_pose_fails_closed_without_full_context() -> None:
    module = _load_module()
    try:
        module._full_d_pose_from_inputs(None, None)
    except ValueError as exc:
        assert "full-video d_pose" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

    assert module._full_d_pose_from_inputs({"component_summary": {"avg_posenet_dist": 4.0}}, None) == 4.0
    assert module._full_d_pose_from_inputs(None, 2.5) == 2.5


def test_vjp_bundle_false_authority_constants() -> None:
    module = _load_module()
    assert module.FALSE_AUTHORITY["score_claim"] is False
    assert module.FALSE_AUTHORITY["promotion_eligible"] is False
    assert module.SCHEMA == "direct_full_scorer_vjp_bundle.v1"


def test_vjp_array_stats_fail_closed_on_nonfinite_gradients() -> None:
    module = _load_module()
    arr = np.array([[[[1.0, np.nan], [np.inf, -3.0]]]], dtype=np.float32)
    stats = module._array_stats(arr)
    assert stats["finite_count"] == 2
    assert stats["nonfinite_count"] == 2
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 1
    blockers = module._gradient_quality_blockers(
        name="segnet_last_rgb",
        stats=stats,
    )
    assert blockers == ["segnet_last_rgb_gradient_nonfinite:2"]
    np.testing.assert_allclose(module._per_pair_l2(arr), np.array([np.sqrt(10.0)], dtype=np.float32))


def test_vjp_gradient_quality_blocks_extreme_finite_metal_drift() -> None:
    module = _load_module()
    blockers = module._gradient_quality_blockers(
        name="segnet_last_rgb",
        stats={"nonfinite_count": 0, "abs_max": 1.0e21},
        max_abs_sanity_limit=1.0e6,
    )
    assert blockers == ["segnet_last_rgb_gradient_abs_max_exceeds_sanity_limit:1e+21>1000000"]


def test_vjp_defaults_to_auto_mlx_gradient_fallback() -> None:
    module = _load_module()
    args = module.build_arg_parser().parse_args(
        [
            "--candidate-cache-dir",
            "/tmp/c",
            "--reference-cache-dir",
            "/tmp/r",
            "--output-dir",
            "/tmp/o",
        ]
    )
    assert args.backend == "mlx"
    assert args.device_type == "auto"
    assert args.max_gradient_abs_sanity_limit == 1.0e6


def test_vjp_storage_preflight_requires_ssd_unless_explicit_local(
    tmp_path: Path,
) -> None:
    module = _load_module()

    payload = module._storage_preflight(
        output_dir=tmp_path / "vjp",
        requested_bytes=1024,
        reserve_free_bytes=0,
        allow_local_output=True,
        storage_plan_path=None,
        cleanup_plan_path=None,
    )
    assert payload["passed"] is True
    assert payload["allow_local_output"] is True
    assert payload["score_claim"] is False

    try:
        module._storage_preflight(
            output_dir=tmp_path / "blocked",
            requested_bytes=1024,
            reserve_free_bytes=0,
            allow_local_output=False,
            storage_plan_path=None,
            cleanup_plan_path=None,
        )
    except SystemExit as exc:
        assert "output_dir_not_on_operator_ssd_tier" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected local-output preflight refusal")


def test_vjp_output_byte_estimate_scales_with_pairs() -> None:
    module = _load_module()

    class Candidate:
        posenet_yuv6_pair = np.zeros((10, 12, 4, 4), dtype=np.float32)
        segnet_last_rgb = np.zeros((10, 3, 4, 4), dtype=np.float32)

    small = module._estimate_vjp_output_bytes(
        candidate=Candidate(),
        start_pair=0,
        stop_pair=1,
        summary_only=False,
    )
    large = module._estimate_vjp_output_bytes(
        candidate=Candidate(),
        start_pair=0,
        stop_pair=10,
        summary_only=False,
    )

    assert large >= small
    assert module._estimate_vjp_output_bytes(
        candidate=Candidate(),
        start_pair=0,
        stop_pair=10,
        summary_only=True,
    ) < large
