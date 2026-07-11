# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tac.boundary_math.island_protection import island_birth_from_signed_np
from tac.boundary_math.persistence_topology_loss import persistence_topology_loss_np
from tac.boundary_math.weight_entropy_penalty_mlx import soft_symbol_entropy_bits_numpy
from tac.cuda_levelset_training import (
    CudaLevelSetConfig,
    TorchLevelSetWitness,
    forward_parity_against_numpy,
    island_birth_from_signed_torch,
    persistence_topology_loss_torch,
    weight_entropy_rate_term_torch,
)


def test_torch_forward_matches_numpy_reference():
    cfg = CudaLevelSetConfig(n_pairs=2, in_feat=9, hidden_dim=8, n_hidden=2, mod_dim=5)
    model = TorchLevelSetWitness.build(cfg, seed=17)
    feats = np.random.default_rng(4).normal(size=(31, 9)).astype(np.float32)
    row = forward_parity_against_numpy(model, feats)
    assert row["argmax_equal"]
    assert row["cosine_phi"] >= 0.9997
    assert row["rgb_max_abs_delta"] <= 5e-5


def test_island_birth_matches_numpy_reference():
    rng = np.random.default_rng(2)
    signed = rng.normal(size=(1, 7, 9)).astype(np.float32)
    weight = (rng.random((1, 7, 9)) > 0.5).astype(np.float32)
    expected = island_birth_from_signed_np(signed, weight, 1.0, form="hinge")
    actual = island_birth_from_signed_torch(
        torch.from_numpy(signed), torch.from_numpy(weight), 1.0, form="hinge"
    )
    assert float(actual) == pytest.approx(expected, abs=2e-6)


def test_persistence_topology_matches_numpy_reference():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(1, 12, 13, 5)).astype(np.float32)
    labels = rng.integers(0, 5, size=(1, 12, 13), dtype=np.int64)
    oh = np.eye(5, dtype=np.float32)[labels]
    expected = persistence_topology_loss_np(logits, oh, (3,), cldice_iters=2)
    actual = persistence_topology_loss_torch(
        torch.from_numpy(logits), torch.from_numpy(labels), (3,), iters=2
    )
    assert float(actual) == pytest.approx(expected, abs=2e-5)


def test_weight_entropy_torch_matches_numpy_per_tensor():
    model = torch.nn.Linear(5, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.linspace(-1.0, 1.0, model.weight.numel()).reshape_as(model.weight))
    bits, _rate = weight_entropy_rate_term_torch(model)
    expected = soft_symbol_entropy_bits_numpy(model.weight.detach().numpy()) * model.weight.numel()
    assert float(bits.detach()) == pytest.approx(expected, rel=2e-5, abs=2e-5)
