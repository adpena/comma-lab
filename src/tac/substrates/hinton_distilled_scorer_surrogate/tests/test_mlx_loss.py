# SPDX-License-Identifier: MIT
"""Canonical Hinton KL T=2.0 ``custom_loss_fn`` unit tests.

These tests cover the loss math + factory + SubstrateAdapterScaffold
integration. They MUST pass before the canonical smoke can be run.

Per CLAUDE.md "MLX portable-local-substrate authority" the tests are
skipped on non-MLX hosts (Linux x86_64 CI) so they remain runnable in the
canonical pytest sweep without breaking the contract on non-Apple-Silicon
machines.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from tac.local_acceleration import EVIDENCE_GRADE_MLX  # noqa: E402
from tac.substrates.hinton_distilled_scorer_surrogate import (  # noqa: E402
    DEFAULT_DISTILLATION_TEMPERATURE,
    DEFAULT_POSE_DIMS,
    DEFAULT_POSE_POOL_GRID,
    DEFAULT_SEGNET_CLASSES,
    HintonMlxCustomLossFnConfig,
    LearnablePoseStudentHead,
    MockTeacherLogitsProvider,
    RealPoseNetTeacherCache,
    build_learnable_pose_student_head,
    build_real_segnet_teacher_cache,
    hinton_distilled_kl_t2_loss,
    kl_divergence_between_softmax,
    make_hinton_custom_loss_fn,
    pose_distillation_mse_loss,
    softmax_with_temperature,
)
from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (  # noqa: E402
    custom_loss_fn_canonical_signature_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[5]


def test_default_temperature_matches_quantizr_canonical_anchor() -> None:
    """CLAUDE.md 'Quantizr intelligence' anchors T=2.0."""

    assert DEFAULT_DISTILLATION_TEMPERATURE == 2.0


def test_default_segnet_classes_matches_upstream() -> None:
    """Upstream smp.Unet(classes=5) per upstream/modules.py."""

    assert DEFAULT_SEGNET_CLASSES == 5


def test_softmax_with_temperature_sums_to_one() -> None:
    logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    out = softmax_with_temperature(logits, 2.0)
    assert abs(float(mx.sum(out).item()) - 1.0) < 1e-5


def test_softmax_with_temperature_rejects_nonpositive_temperature() -> None:
    logits = mx.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        softmax_with_temperature(logits, 0.0)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        softmax_with_temperature(logits, -1.0)


def test_kl_self_is_zero() -> None:
    logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    sm = softmax_with_temperature(logits, 2.0)
    kl = kl_divergence_between_softmax(sm, sm)
    assert abs(float(kl.item())) < 1e-5


def test_hinton_kl_self_is_zero() -> None:
    logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    loss = hinton_distilled_kl_t2_loss(logits, logits, temperature=2.0)
    assert abs(float(loss.item())) < 1e-5


def test_hinton_kl_nonzero_for_different_logits() -> None:
    a = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    b = mx.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    loss = hinton_distilled_kl_t2_loss(a, b, temperature=2.0)
    assert float(loss.item()) > 0.0


def test_hinton_kl_t2_scaling_canonical() -> None:
    """T**2 factor (Hinton 2014 §2.1) makes higher T → larger loss."""

    a = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    b = mx.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    l1 = float(hinton_distilled_kl_t2_loss(a, b, temperature=1.0).item())
    l2 = float(hinton_distilled_kl_t2_loss(a, b, temperature=2.0).item())
    l4 = float(hinton_distilled_kl_t2_loss(a, b, temperature=4.0).item())
    # With T**2 scaling, increasing T amplifies the loss magnitude
    # because the soft-label gradient scales as 1/T**2 and the loss is
    # multiplied by T**2.
    assert l1 < l2 < l4, f"T**2 scaling violated: {l1} {l2} {l4}"


def test_hinton_kl_rejects_nonpositive_temperature() -> None:
    a = mx.array([[1.0, 2.0]])
    b = mx.array([[2.0, 1.0]])
    with pytest.raises(ValueError, match="temperature must be > 0"):
        hinton_distilled_kl_t2_loss(a, b, temperature=0.0)


def test_pose_distillation_mse_loss_matches_continuous_pose_mse() -> None:
    student = mx.array([[1.0, 2.0, 3.0], [0.0, 0.5, 1.0]])
    teacher = mx.array([[1.0, 0.0, 3.0], [1.0, 0.5, 2.0]])
    loss = pose_distillation_mse_loss(student, teacher)
    assert float(loss.item()) == pytest.approx(
        (0.0 + 4.0 + 0.0 + 1.0 + 0.0 + 1.0) / 6.0
    )


def test_pose_distillation_mse_loss_supports_per_dim_scale() -> None:
    student = mx.array([[2.0, 4.0]])
    teacher = mx.array([[0.0, 0.0]])
    loss = pose_distillation_mse_loss(
        student,
        teacher,
        per_dim_scale=mx.array([2.0, 4.0]),
    )
    assert float(loss.item()) == pytest.approx(1.0)


def test_real_posenet_teacher_cache_indexes_pair_pose() -> None:
    cache = RealPoseNetTeacherCache(
        teacher_pose_np=mx.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        ),
        num_pairs=2,
        pose_dims=3,
        per_dim_scale=mx.array([1.0, 2.0, 3.0]),
        upstream_posenet_safetensors_sha256="a" * 64,
        cache_build_seconds=1.25,
    )
    out = cache.teacher_pose_for_indices(mx.array([1, 0], dtype=mx.int32))
    assert out.shape == (2, 3)
    assert tuple(cache.per_dim_scale.shape) == (3,)
    assert cache.upstream_posenet_safetensors_sha256 == "a" * 64
    assert cache.cache_build_seconds == pytest.approx(1.25)


def test_real_posenet_teacher_cache_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="teacher_pose_np must have shape"):
        RealPoseNetTeacherCache(
            teacher_pose_np=mx.zeros((2, 4)),
            num_pairs=2,
            pose_dims=3,
        )


def test_real_posenet_teacher_cache_rejects_scale_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="per_dim_scale must have shape"):
        RealPoseNetTeacherCache(
            teacher_pose_np=mx.zeros((2, 3)),
            num_pairs=2,
            pose_dims=3,
            per_dim_scale=mx.zeros((4,)),
        )


def test_build_learnable_pose_student_head_returns_canonical_shape() -> None:
    assert DEFAULT_POSE_DIMS == 6
    assert DEFAULT_POSE_POOL_GRID == 4
    head = build_learnable_pose_student_head(seed=7)
    assert head.weight.shape == (2 * 4 * 4 * 3, 6)
    assert head.bias.shape == (6,)
    rgb_0 = mx.ones((2, 8, 8, 3))
    rgb_1 = mx.zeros((2, 8, 8, 3))
    out = head(rgb_0, rgb_1)
    assert out.shape == (2, 6)


def test_learnable_pose_student_head_forward_with_params_matches_call() -> None:
    head = build_learnable_pose_student_head(pose_dims=2, pool_grid=2, seed=3)
    rgb_0 = mx.ones((1, 4, 4, 3))
    rgb_1 = mx.zeros((1, 4, 4, 3))
    out_call = head(rgb_0, rgb_1)
    out_params = head.forward_with_params(
        rgb_0,
        rgb_1,
        {"weight": head.weight, "bias": head.bias},
    )
    assert float(mx.max(mx.abs(out_call - out_params)).item()) < 1e-7


def test_learnable_pose_student_head_rejects_too_small_spatial_grid() -> None:
    head = LearnablePoseStudentHead(
        weight=mx.zeros((2 * 4 * 4 * 3, 6)),
        bias=mx.zeros((6,)),
        pose_dims=6,
        pool_grid=4,
    )
    with pytest.raises(ValueError, match="too small"):
        head(mx.ones((1, 3, 8, 3)), mx.ones((1, 3, 8, 3)))


def test_mock_teacher_logits_shape() -> None:
    import numpy as np

    mock = MockTeacherLogitsProvider(
        num_classes=5,
        spatial_downsample_factor=4,
    )
    frames = mx.array(
        np.random.RandomState(42).rand(2, 16, 16, 3).astype(np.float32)
    )
    out = mock.teacher_logits(frames)
    assert out.shape == (2, 4, 4, 5)


def test_mock_teacher_logits_is_deterministic() -> None:
    """Same input frames → same output logits across invocations."""

    import numpy as np

    mock = MockTeacherLogitsProvider(num_classes=5, spatial_downsample_factor=4)
    frames = mx.array(
        np.random.RandomState(0).rand(1, 8, 8, 3).astype(np.float32)
    )
    out1 = mock.teacher_logits(frames)
    out2 = mock.teacher_logits(frames)
    assert mx.all(out1 == out2).item()


def test_mock_teacher_logits_distinct_per_class() -> None:
    """Each class produces distinct logits values (KL non-degenerate)."""

    import numpy as np

    mock = MockTeacherLogitsProvider(num_classes=5, spatial_downsample_factor=4)
    frames = mx.array(
        np.random.RandomState(1).rand(1, 8, 8, 3).astype(np.float32)
    )
    out = mock.teacher_logits(frames)
    # Standard deviation across classes axis should be nonzero.
    std_per_pixel = mx.std(out, axis=-1)
    assert float(mx.mean(std_per_pixel).item()) > 1e-3


def test_mock_teacher_rejects_indivisible_spatial_dim() -> None:
    import numpy as np

    mock = MockTeacherLogitsProvider(num_classes=5, spatial_downsample_factor=4)
    # 7 is not divisible by 4.
    frames = mx.array(
        np.random.RandomState(0).rand(1, 7, 8, 3).astype(np.float32)
    )
    with pytest.raises(ValueError, match="divisible by"):
        mock.teacher_logits(frames)


def test_mock_teacher_rejects_invalid_num_classes() -> None:
    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        MockTeacherLogitsProvider(num_classes=1)


def test_mock_teacher_rejects_invalid_downsample() -> None:
    with pytest.raises(ValueError, match="spatial_downsample_factor must be >= 1"):
        MockTeacherLogitsProvider(spatial_downsample_factor=0)


def test_hinton_config_default_values() -> None:
    cfg = HintonMlxCustomLossFnConfig()
    assert cfg.distillation_weight == 0.5
    assert cfg.temperature == 2.0
    assert cfg.student_head_out_channels == 5
    assert cfg.evidence_grade == EVIDENCE_GRADE_MLX


def test_hinton_config_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="distillation_weight must be >= 0"):
        HintonMlxCustomLossFnConfig(distillation_weight=-0.1)


def test_hinton_config_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be > 0"):
        HintonMlxCustomLossFnConfig(temperature=0.0)


def test_hinton_config_rejects_invalid_num_classes() -> None:
    with pytest.raises(ValueError, match="student_head_out_channels must be >= 2"):
        HintonMlxCustomLossFnConfig(student_head_out_channels=1)


def test_hinton_config_rejects_wrong_evidence_grade() -> None:
    """CLAUDE.md 'MLX portable-local-substrate authority' non-negotiable."""

    with pytest.raises(ValueError, match="evidence_grade must be"):
        HintonMlxCustomLossFnConfig(evidence_grade="[contest-CUDA]")
    with pytest.raises(ValueError, match="evidence_grade must be"):
        HintonMlxCustomLossFnConfig(evidence_grade="[contest-CPU]")
    with pytest.raises(ValueError, match="evidence_grade must be"):
        HintonMlxCustomLossFnConfig(evidence_grade="[macOS-CPU advisory]")


def test_make_hinton_custom_loss_fn_signature() -> None:
    """Returned callable matches SubstrateAdapterScaffold.custom_loss_fn signature."""

    fn = make_hinton_custom_loss_fn()
    # Three positional args: bundle, indices, targets_batch.
    import inspect

    sig = inspect.signature(fn)
    assert len(sig.parameters) == 3


def test_make_hinton_custom_loss_fn_returns_scalar_loss() -> None:
    """End-to-end loss computation with a fake bundle."""

    import numpy as np

    class _FakeBundle:
        def __call__(self, indices: object) -> object:
            # Return canonical Slot 1 shape: (B, 2, 3, H, W) in [0, 255].
            # Use small spatial dims divisible by 4.
            n = int(indices.shape[0])  # type: ignore[attr-defined]
            arr = np.random.RandomState(2).rand(n, 2, 3, 16, 16).astype(np.float32) * 255.0
            return mx.array(arr)

    bundle = _FakeBundle()
    fn = make_hinton_custom_loss_fn(
        HintonMlxCustomLossFnConfig(
            teacher_provider=MockTeacherLogitsProvider(
                num_classes=5,
                spatial_downsample_factor=4,
            ),
        )
    )
    targets = mx.array(
        np.random.RandomState(3).rand(2, 16, 16, 3).astype(np.float32)
    )
    indices = mx.array([0, 1], dtype=mx.int32)
    loss = fn(bundle, indices, targets)
    assert loss.shape == ()
    val = float(loss.item())
    assert val == val  # not NaN
    assert val > 0.0


def test_make_hinton_custom_loss_fn_distillation_term_is_nonzero() -> None:
    """The KL branch must not collapse to self-KL on decoded frames."""

    import numpy as np

    decoded_arr = np.random.RandomState(4).rand(2, 2, 3, 16, 16).astype(np.float32) * 255.0
    targets = mx.array(np.random.RandomState(5).rand(2, 16, 16, 3).astype(np.float32))
    indices = mx.array([0, 1], dtype=mx.int32)

    class _FakeBundle:
        def __call__(self, _indices: object) -> object:
            return mx.array(decoded_arr)

    provider = MockTeacherLogitsProvider(num_classes=5, spatial_downsample_factor=4)
    mse_only = make_hinton_custom_loss_fn(
        HintonMlxCustomLossFnConfig(
            distillation_weight=0.0,
            teacher_provider=provider,
        )
    )
    with_distill = make_hinton_custom_loss_fn(
        HintonMlxCustomLossFnConfig(
            distillation_weight=0.5,
            teacher_provider=provider,
        )
    )

    base = float(mse_only(_FakeBundle(), indices, targets).item())
    distill = float(with_distill(_FakeBundle(), indices, targets).item())

    assert distill > base


def test_real_segnet_teacher_cache_uses_upstream_rgb_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-teacher cache must match upstream 0..255 SegNet input scale."""

    import numpy as np
    import torch

    observed: dict[str, float] = {}

    class _FakeSegNet:
        def eval(self) -> None:
            observed["eval_called"] = 1.0

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            observed["input_max"] = float(x.max().item())
            observed["input_min"] = float(x.min().item())
            return x[:, -1, ...]

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            b, _c, h, w = x.shape
            return torch.zeros((b, 5, h, w), dtype=torch.float32)

    def _fake_load_default_scorers(_upstream_dir: object, *, device: str) -> tuple[None, _FakeSegNet]:
        observed["device_is_cpu"] = 1.0 if device == "cpu" else 0.0
        return None, _FakeSegNet()

    import tac.scorer

    monkeypatch.setattr(tac.scorer, "load_default_scorers", _fake_load_default_scorers)
    frames = np.array(
        [
            [[[0, 128, 255], [64, 32, 16]], [[255, 1, 2], [3, 4, 5]]],
            [[[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]],
        ],
        dtype=np.uint8,
    )

    cache = build_real_segnet_teacher_cache(frames, upstream_dir="upstream", device="cpu")

    assert observed["eval_called"] == 1.0
    assert observed["device_is_cpu"] == 1.0
    assert observed["input_max"] == 255.0
    assert observed["input_min"] == 0.0
    assert cache.frame_count == 2
    assert cache.height == 2
    assert cache.width == 2
    assert cache.num_classes == 5


def test_hinton_smoke_plan_records_real_segnet_effective_downsample_and_provenance(
    tmp_path: Path,
) -> None:
    output_report = tmp_path / "plan.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_hinton_mlx_long_training_smoke.py"),
            "--output-report",
            str(output_report),
            "--teacher-provider",
            "real_segnet",
            "--teacher-cache-device",
            "cpu",
            "--max-frames",
            "4",
            "--smoke-epochs",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    stdout = json.loads(result.stdout)
    report = json.loads(output_report.read_text(encoding="utf-8"))

    assert stdout["mode"] == "plan_only"
    assert report["teacher_provider"] == "real_segnet"
    assert report["teacher_cache_device"] == "cpu"
    assert report["spatial_downsample_factor"] == 4
    assert report["effective_spatial_downsample_factor"] == 1
    assert report["effective_student_head_spatial_downsample_factor"] == 1
    assert report["effective_teacher_spatial_downsample_factor"] == 1
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["score_claim"] is False
    provenance = report["repo_provenance"]
    assert provenance["schema"] == "hinton_mlx_smoke_repo_provenance.v1"
    assert len(provenance["command_sha256"]) == 64
    assert provenance["score_claim"] is False
    assert "tools/run_hinton_mlx_long_training_smoke.py" in provenance["file_sha256"]


def test_custom_loss_fn_canonical_signature_hash_stable() -> None:
    """Sister-cascade signature hash for cross-wave verification."""

    h1 = custom_loss_fn_canonical_signature_hash()
    h2 = custom_loss_fn_canonical_signature_hash()
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_substrate_adapter_scaffold_can_carry_our_custom_loss_fn() -> None:
    """The factory output is accepted by SubstrateAdapterScaffold.custom_loss_fn."""

    from tac.local_acceleration.pr95_hnerv_mlx_long_training import (
        SubstrateAdapterScaffold,
    )

    fn = make_hinton_custom_loss_fn()
    scaffold = SubstrateAdapterScaffold(
        candidate_id="test_hinton_distilled_scorer_surrogate",
        candidate_class_shift_paradigm="hinton_distilled_scorer_surrogate",
        custom_loss_fn=fn,
    )
    assert scaffold.custom_loss_fn is fn
    assert scaffold.as_dict()["has_custom_loss_fn"] is True
