import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tac.cuda_v9_island_runtime import (
    TorchIslandTargetRuntime,
    birth_scaled_logit_offsets,
)


def _fixture():
    labels = np.zeros((2, 9, 11), dtype=np.int64)
    labels[:, 4, 2:9] = 1
    labels[:, 2:5, 7:10] = 3
    flags = {
        "--amplify-persist": "uniform",
        "--ladder-island-homotopy": True,
        "--seed-island-eased": True,
        "--island-dilate-px": 2,
        "--seed-blend": 0.75,
        "--containment-mode": "shield",
        "--containment-damp": 0.1,
    }
    return labels, flags


def test_ladder_refresh_keeps_static_storage_and_partition():
    labels, flags = _fixture()
    runtime = TorchIslandTargetRuntime(
        labels, lane_cls=1, movable_cls=3, flags=flags, device="cpu"
    )
    pointers = (runtime.weight.data_ptr(), runtime.lane_mask.data_ptr(), runtime.movable_mask.data_ptr())
    assert runtime.refresh_amplify_(lane_px=1, movable_px=0)
    assert pointers == (
        runtime.weight.data_ptr(),
        runtime.lane_mask.data_ptr(),
        runtime.movable_mask.data_ptr(),
    )
    support = runtime.weight > 0
    assert torch.equal((runtime.lane_mask > 0) | (runtime.movable_mask > 0), support)
    assert not bool(((runtime.lane_mask > 0) & (runtime.movable_mask > 0)).any())
    assert not runtime.refresh_amplify_(lane_px=1, movable_px=0)


def test_seed_is_training_only_gt_appearance_residual():
    labels, flags = _fixture()
    runtime = TorchIslandTargetRuntime(
        labels, lane_cls=1, movable_cls=3, flags=flags, device="cpu"
    )
    gt = np.random.default_rng(7).normal(size=(*labels.shape, 3)).astype(np.float32)
    seed = runtime.build_protected_seed(gt)
    expected = torch.from_numpy(gt) * 0.75
    support = seed.mask[..., None].expand_as(seed.residual)
    assert torch.allclose(seed.residual.detach()[support], expected[support])
    assert torch.count_nonzero(seed.residual.detach()[~support]) == 0


def test_seed_resizes_native_rgb_to_scorer_grid_before_masking():
    labels, flags = _fixture()
    runtime = TorchIslandTargetRuntime(
        labels, lane_cls=1, movable_cls=3, flags=flags, device="cpu"
    )
    gt = np.ones((2, 18, 22, 3), dtype=np.float32) * 8.0
    seed = runtime.build_protected_seed(gt)
    assert seed.residual.shape == (2, 9, 11, 3)
    assert torch.all(seed.residual.detach()[seed.mask] == 6.0)


def test_logit_offsets_scale_only_watched_classes():
    base = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    actual = birth_scaled_logit_offsets(base, {1: 0.25, 3: 0.5})
    assert torch.equal(actual, torch.tensor([0.0, 0.25, 2.0, 1.5, 4.0]))
