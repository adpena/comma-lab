# SPDX-License-Identifier: MIT
"""MLX-bound tests: score-aware loss + adapter + end-to-end harness run.

MLX-bound tests skip cleanly on non-Apple-Silicon CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.substrates._shared.mlx_score_aware import (
    MlxScoreAwareAdapter,
    RendererBundle,
    decode_frames_nhwc01,
    run_mlx_score_aware_full_main,
    score_aware_loss,
)

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required (Apple Silicon)")

_LANE = "lane_mlx_score_aware_harness_refactor_plus_4_unlock_20260527"


def _tiny_dreamer_bundle(num_pairs: int = 4, distill: float = 0.5) -> RendererBundle:
    import mlx.core as mx

    from tac.substrates.dreamer_v3_rssm.module import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )

    cfg = DreamerV3RSSMConfig(
        num_pairs=num_pairs,
        num_groups=2,
        num_categories=4,
        decoder_latent_dim=8,
        base_channels=4,
        eval_size=(384, 512),
    )
    model = DreamerV3RSSMSubstrateMLX(cfg)
    t0 = mx.zeros((num_pairs, 384, 512, 3))
    t1 = mx.zeros((num_pairs, 384, 512, 3))
    # When a distillation term is active these helper bundles use the
    # scorer-BLIND mock (no real SegNet staged in the unit-test fast path), so
    # they must EXPLICITLY opt in via allow_mock_scorer_teacher per the C6 IBPS
    # fail-closed invariant. The real-scorer-bound path is exercised in
    # test_scorer_binding.py with a real SegNet teacher.
    return RendererBundle(
        model=model,
        target_rgb_0=t0,
        target_rgb_1=t1,
        num_pairs=num_pairs,
        forward_convention="call_b2chw_255",
        distillation_weight=distill,
        allow_mock_scorer_teacher=distill > 0.0,
    )


# --------------------------------------------------------------------------- #
# loss module
# --------------------------------------------------------------------------- #


@mlx_only
def test_decode_frames_nhwc01_shapes() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle()
    idx = mx.array([0, 1], dtype=mx.int32)
    rgb_0, rgb_1 = decode_frames_nhwc01(bundle, idx)
    assert rgb_0.shape == (2, 384, 512, 3)
    assert rgb_1.shape == (2, 384, 512, 3)


@mlx_only
def test_score_aware_loss_is_finite_and_decomposed() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.5)
    idx = mx.array([0, 1], dtype=mx.int32)
    total, parts = score_aware_loss(bundle, idx)
    mx.eval(total)
    assert float(total.item()) == float(total.item())  # not NaN
    assert {"recon", "distill", "total"} <= set(parts)


@mlx_only
def test_score_aware_loss_no_distill_when_weight_zero() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.0)
    idx = mx.array([0, 1], dtype=mx.int32)
    _total, parts = score_aware_loss(bundle, idx)
    assert "distill" not in parts
    assert "recon" in parts


@mlx_only
def test_score_aware_loss_extra_term_weighted() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.0)
    bundle.extra_loss_terms = lambda _m, _i: {"commit": mx.array(2.0)}
    bundle.extra_loss_weights = {"commit": 0.5}
    idx = mx.array([0], dtype=mx.int32)
    _total, parts = score_aware_loss(bundle, idx)
    assert "commit" in parts
    mx.eval(parts["commit"])
    assert abs(float(parts["commit"].item()) - 2.0) < 1e-5


# --------------------------------------------------------------------------- #
# adapter module
# --------------------------------------------------------------------------- #


@mlx_only
def test_adapter_train_step_reduces_loss_over_steps() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.0)
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.array([0, 1, 2, 3], dtype=mx.int32)
    losses = [
        adapter.train_step(batch, learning_rate=1e-2, loss_weights={})["total"]
        for _ in range(20)
    ]
    assert losses[-1] < losses[0]


@mlx_only
def test_adapter_trains_pose_head_jointly() -> None:
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate import (
        RealPoseNetTeacherCache,
        build_learnable_pose_student_head,
    )

    base = _tiny_dreamer_bundle(num_pairs=4, distill=0.0)
    pose_teacher = RealPoseNetTeacherCache(
        teacher_pose_np=mx.ones((4, 6)),
        num_pairs=4,
        pose_dims=6,
    )
    pose_head = build_learnable_pose_student_head(seed=11)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        pose_distillation_weight=0.5,
        pose_scorer_teacher=pose_teacher,
        learnable_pose_student_head=pose_head,
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.array([0, 1, 2, 3], dtype=mx.int32)
    w0 = mx.array(pose_head.weight)
    adapter.train_step(batch, learning_rate=1e-2, loss_weights={})
    moved = float(mx.max(mx.abs(pose_head.weight - w0)).item())
    assert moved > 0.0, "pose head params must train jointly (sibling step)"


@mlx_only
def test_adapter_satisfies_protocol() -> None:
    from tac.training.long_training_canonical import validate_substrate_adapter

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    validate_substrate_adapter(adapter)


@mlx_only
def test_adapter_optimizer_step_raises_style_a_stub() -> None:
    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    with pytest.raises(NotImplementedError, match="Style B train_step"):
        adapter.optimizer_step(adapter.model, None, 1e-3)


@mlx_only
def test_adapter_export_state_dict_writes_portable_npsd(tmp_path: Path) -> None:
    from tac.substrates._shared.numpy_portable_inflate import (
        unpack_state_dict_numpy,
    )

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    target = tmp_path / "ckpt.state"
    adapter.export_state_dict(adapter.model, target)
    blob = target.with_suffix(target.suffix + ".npsd")
    assert blob.is_file()
    restored = unpack_state_dict_numpy(blob.read_bytes())
    assert len(restored) > 0


@mlx_only
def test_adapter_score_aware_components_pure_recon_returns_none() -> None:
    """Pure-reconstruction mode preserves legacy None contract.

    PER_AXIS_DECOMPOSITION GAP FIX 2026-05-28: sister-adapter parity preserved
    when neither scorer surrogate is active. The fix MUST NOT synthesize
    scorer-unbound per-axis rows.
    """
    import mlx.core as mx

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(distill=0.0), substrate_id="dreamer_v3_rssm"
    )
    assert adapter.score_aware_components(adapter.model, mx.array([0])) is None


@mlx_only
def test_adapter_score_aware_components_seg_bound_populates_per_axis() -> None:
    """Hinton-distilled scorer-bound surrogate populates per-axis per Catalog #356.

    PER_AXIS_DECOMPOSITION GAP FIX 2026-05-28 per Z6-v2 + Hinton + 600-pair
    Contrarian VETO op-routable #4: when ``distillation_weight > 0`` (SegNet
    teacher wired via mock or real) the per-axis decomposition MUST emit
    seg+recon_aux+archive_bytes rows so the canonical
    ``AxisDecomposition.from_dict`` round-trip works at the downstream
    cathedral ranker boundary.
    """
    import mlx.core as mx

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(distill=0.5), substrate_id="dreamer_v3_rssm"
    )
    out = adapter.score_aware_components(adapter.model, mx.array([0, 1], dtype=mx.int32))
    assert out is not None
    assert "seg" in out
    assert "pose" in out  # 0.0 (no pose teacher in mock fixture) per fail-closed
    assert "recon_aux" in out
    assert "archive_bytes" in out
    # All values finite per AxisDecomposition __post_init__ invariant.
    for key, value in out.items():
        assert isinstance(value, float), f"{key}={value!r} must be float"
        assert value == value, f"{key}={value!r} must not be NaN"
    # seg > 0 because Hinton-KL is non-negative + the bundle has distill=0.5.
    assert out["seg"] >= 0.0
    # pose = 0.0 because the mock fixture does not wire a pose teacher
    # (no pose_distill component emitted by score_aware_loss).
    assert out["pose"] == 0.0
    # archive_bytes = 0.0 per AxisDecomposition NaN-safe rule (per-step
    # delta undefined at MLX L2; archive built post-training).
    assert out["archive_bytes"] == 0.0


@mlx_only
def test_adapter_score_aware_components_both_teachers_populates_seg_and_pose() -> None:
    """Both SegNet + PoseNet teachers wired → per-axis seg AND pose populated.

    PER_AXIS_DECOMPOSITION GAP FIX 2026-05-28 cross-family sister: the
    canonical scorer-bound BOTH-TEACHER-WIRED contract (Catalog #164) IS
    the surface where the GAP closed. Cross-family seg/pose attribution
    becomes possible only when BOTH axes are populated empirically.
    """
    import mlx.core as mx

    from tac.substrates.hinton_distilled_scorer_surrogate import (
        RealPoseNetTeacherCache,
        build_learnable_pose_student_head,
    )

    base = _tiny_dreamer_bundle(num_pairs=4, distill=0.5)
    pose_teacher = RealPoseNetTeacherCache(
        teacher_pose_np=mx.ones((4, 6)),
        num_pairs=4,
        pose_dims=6,
    )
    pose_head = build_learnable_pose_student_head(seed=17)
    bundle = RendererBundle(
        model=base.model,
        target_rgb_0=base.target_rgb_0,
        target_rgb_1=base.target_rgb_1,
        num_pairs=base.num_pairs,
        forward_convention=base.forward_convention,
        distillation_weight=0.5,
        allow_mock_scorer_teacher=True,  # seg side via mock (no real SegNet here)
        pose_distillation_weight=0.5,
        pose_scorer_teacher=pose_teacher,
        learnable_pose_student_head=pose_head,
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    out = adapter.score_aware_components(
        adapter.model, mx.array([0, 1, 2, 3], dtype=mx.int32)
    )
    assert out is not None
    assert out["seg"] >= 0.0
    assert out["pose"] >= 0.0  # pose-MSE is non-negative
    assert out["recon_aux"] >= 0.0
    assert out["archive_bytes"] == 0.0


@mlx_only
def test_adapter_score_aware_components_compatible_with_axis_decomposition() -> None:
    """Per-axis dict round-trips through canonical AxisDecomposition contract.

    Catalog #356 STRICT preflight gate contract: the per-axis surface MUST
    map directly into ``AxisDecomposition.from_dict``-like consumption. The
    canonical mapping is: seg → predicted_d_seg_delta; pose →
    predicted_d_pose_delta; archive_bytes → predicted_archive_bytes_delta.
    """
    import mlx.core as mx

    from tac.cathedral.consumer_contract import AxisDecomposition
    from tac.provenance import (
        build_provenance_for_predicted,
        provenance_to_dict,
    )

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(distill=0.5), substrate_id="dreamer_v3_rssm"
    )
    out = adapter.score_aware_components(
        adapter.model, mx.array([0, 1], dtype=mx.int32)
    )
    assert out is not None
    # Build a canonical AxisDecomposition from the per-axis dict + canonical
    # Provenance per Catalog #323. This verifies the GAP-fix output integrates
    # with the downstream cathedral ranker boundary contract.
    prov = build_provenance_for_predicted(
        model_id="mlx_score_aware_per_axis_decomposition_v1",
        inputs_sha256="a" * 64,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="macos_arm64",
    )
    decomp = AxisDecomposition(
        predicted_d_seg_delta=out["seg"],
        predicted_d_pose_delta=out["pose"],
        predicted_archive_bytes_delta=int(out["archive_bytes"]),
        axis_tag="[predicted]",
        canonical_provenance=provenance_to_dict(prov),
    )
    # Round-trip stable (no NaN, no infinite, no type rejection).
    d = decomp.as_dict()
    assert d["predicted_d_seg_delta"] == out["seg"]
    assert d["predicted_d_pose_delta"] == out["pose"]
    assert d["predicted_archive_bytes_delta"] == 0
    assert d["axis_tag"] == "[predicted]"
    assert d["canonical_provenance"]["measurement_axis"] == "[macOS-MLX research-signal]"


# --------------------------------------------------------------------------- #
# harness orchestrator (end-to-end through canonical run_long_training)
# --------------------------------------------------------------------------- #


@mlx_only
def test_run_mlx_score_aware_full_main_end_to_end(tmp_path: Path) -> None:
    bundle = _tiny_dreamer_bundle(num_pairs=4, distill=0.5)
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="dreamer_v3_rssm",
        lane_id=_LANE,
        output_dir=tmp_path / "run",
        epochs=3,
        batch_pair_indices_per_step=2,
        learning_rate=1e-3,
        seed=0,
        notes="harness refactor end-to-end: dreamer renderer + zero targets",
    )
    assert artifact.total_epochs_completed == 3
    assert artifact.promotable is False
    d = artifact.as_dict()
    assert d.get("score_claim") is False
    assert d.get("promotion_eligible") is False


@mlx_only
def test_run_verifies_inflate_portability_fails_closed(tmp_path: Path) -> None:
    from tac.substrates._shared.mlx_score_aware import MlxScoreAwareHarnessError

    bad_inflate = tmp_path / "inflate.py"
    bad_inflate.write_text("import torch\n", encoding="utf-8")
    bundle = _tiny_dreamer_bundle(num_pairs=4, distill=0.0)
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        run_mlx_score_aware_full_main(
            bundle=bundle,
            substrate_id="dreamer_v3_rssm",
            lane_id=_LANE,
            output_dir=tmp_path / "run",
            epochs=1,
            batch_pair_indices_per_step=2,
            inflate_py_path=bad_inflate,
            notes="inflate portability fail-closed before training",
        )
