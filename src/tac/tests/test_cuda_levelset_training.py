# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tac.boundary_math.island_protection import island_birth_from_signed_np
from tac.boundary_math.persistence_topology_loss import persistence_topology_loss_np
from tac.boundary_math.weight_entropy_penalty_mlx import soft_symbol_entropy_bits_numpy
from tac.cuda_levelset_training import (
    CudaGraphRecaptureGuard,
    CudaLevelSetConfig,
    DeterministicPairCursor,
    TorchExecutionPolicy,
    TorchPoseCarrier,
    TorchLevelSetWitness,
    apply_torch_execution_policy,
    area_constraint_torch,
    chroma_boundary_loss,
    clip_grad_groups,
    compile_identity_probe,
    eikonal_and_length,
    forward_parity_against_numpy,
    island_birth_from_signed_torch,
    island_birth_perclass_from_signed_torch,
    parameter_groups,
    persistence_topology_loss_torch,
    pose_objective_torch,
    select_torch_execution_policy,
    structured_sdf_prefit,
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


@pytest.mark.parametrize("form", ["hinge", "softplus"])
def test_perclass_island_birth_unit_multipliers_recover_combined_term(form):
    gen = torch.Generator().manual_seed(23)
    signed = torch.randn(2, 7, 9, generator=gen)
    support = torch.rand(2, 7, 9, generator=gen) > 0.35
    partition = torch.rand(2, 7, 9, generator=gen) > 0.5
    mask_a = support & partition
    mask_b = support & ~partition
    weight = support.to(torch.float32) * (0.2 + torch.rand(2, 7, 9, generator=gen))
    combined = island_birth_from_signed_torch(signed, weight, 0.8, form=form)
    perclass = island_birth_perclass_from_signed_torch(
        signed, weight, mask_a, mask_b, 0.8, 1.0, 1.0, form=form
    )
    assert torch.allclose(perclass, combined, atol=2e-6, rtol=2e-6)


def test_birth_completion_scales_only_persistence_recall_not_cldice():
    gen = torch.Generator().manual_seed(29)
    logits = torch.randn(1, 12, 13, 5, generator=gen)
    labels = torch.randint(0, 5, (1, 12, 13), generator=gen)
    no_recall = persistence_topology_loss_torch(
        logits, labels, (1, 3), iters=2, recall_weight=0.0
    )
    zero_scaled = persistence_topology_loss_torch(
        logits, labels, (1, 3), iters=2, recall_class_scale=(0.0, 0.0)
    )
    assert torch.allclose(zero_scaled, no_recall, atol=2e-6, rtol=2e-6)


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


def test_batched_nonlinear_losses_equal_exact_mean_of_serial_pair_losses():
    gen = torch.Generator().manual_seed(41)
    logits = torch.randn(2, 9, 11, 5, generator=gen)
    labels = torch.randint(0, 5, (2, 9, 11), generator=gen)
    signed = torch.randn(2, 9, 11, generator=gen)
    weight = torch.rand(2, 9, 11, generator=gen)
    weight[1, :6] = 0.0  # unequal denominators expose an accidental global reduction
    rgb = torch.rand(2, 9, 11, 3, generator=gen)
    gt = torch.rand(2, 9, 11, 3, generator=gen)
    ann = torch.rand(2, 9, 11, generator=gen) > 0.35
    phi = torch.randn(2, 9, 11, 5, generator=gen)
    pose = torch.randn(2, 6, generator=gen)
    pose_target = torch.randn(2, 6, generator=gen)

    checks = (
        lambda sl: island_birth_from_signed_torch(signed[sl], weight[sl], 1.0),
        lambda sl: area_constraint_torch(logits[sl], labels[sl], {1: 0.7, 3: 1.2}),
        lambda sl: persistence_topology_loss_torch(logits[sl], labels[sl], (1, 3)),
        lambda sl: chroma_boundary_loss(rgb[sl], gt[sl], ann[sl]),
        lambda sl: eikonal_and_length(phi[sl])[0],
        lambda sl: eikonal_and_length(phi[sl])[1],
        lambda sl: pose_objective_torch(pose[sl], pose_target[sl]),
    )
    for fn in checks:
        batched = fn(slice(None))
        serial_mean = torch.stack([fn(slice(i, i + 1)) for i in range(2)]).mean()
        assert torch.allclose(batched, serial_mean, atol=2e-6, rtol=2e-6)


def test_weight_entropy_torch_matches_numpy_per_tensor():
    model = torch.nn.Linear(5, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.linspace(-1.0, 1.0, model.weight.numel()).reshape_as(model.weight))
    bits, _rate = weight_entropy_rate_term_torch(model)
    expected = soft_symbol_entropy_bits_numpy(model.weight.detach().numpy()) * model.weight.numel()
    assert float(bits.detach()) == pytest.approx(expected, rel=2e-5, abs=2e-5)


def test_pose_carrier_matches_canonical_numpy_warp_and_dxi_receives_gradient():
    from tac.boundary_math.warp_real_luma_frame0 import (
        GroundHomographyGeom,
        warp_frame0_native_numpy,
    )

    geom = GroundHomographyGeom.eon(native_hw=(12, 16), pitch=-0.01)
    xi = np.array([[0.002, -0.001, 0.003, 0.0004, -0.0002, 0.0003]], np.float32)
    src_np = np.linspace(0.0, 255.0, 12 * 16 * 3, dtype=np.float32).reshape(1, 12, 16, 3)
    carrier = TorchPoseCarrier.build(xi, geom)
    actual = carrier(torch.from_numpy(src_np), torch.tensor([0]))
    expected = warp_frame0_native_numpy(src_np[0], xi[0], geom, compute_dtype=np.float32)
    assert np.max(np.abs(actual.detach().numpy()[0] - expected)) < 2e-2
    actual.square().mean().backward()
    assert carrier.dxi.grad is not None
    assert float(carrier.dxi.grad.abs().sum()) > 0.0
    clone = TorchPoseCarrier.build(xi, geom)
    clone.load_state_dict(carrier.state_dict())
    assert torch.equal(clone.dxi, carrier.dxi)


def test_pose_carrier_zero_twist_has_finite_identity_forward_and_gradient():
    from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom

    geom = GroundHomographyGeom.eon(native_hw=(8, 10), pitch=0.0)
    carrier = TorchPoseCarrier.build(np.zeros((1, 6), np.float32), geom)
    src = torch.linspace(0.0, 255.0, 8 * 10 * 3).reshape(1, 8, 10, 3)
    actual = carrier(src, torch.tensor([0]))
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, src, atol=2e-4)
    actual.square().mean().backward()
    assert carrier.dxi.grad is not None
    assert torch.isfinite(carrier.dxi.grad).all()
    assert float(carrier.dxi.grad.abs().sum()) > 0.0


def test_structured_prefit_changes_shared_sdf_trunk_reduces_loss_and_freezes_code():
    cfg = CudaLevelSetConfig(n_pairs=2, in_feat=7, hidden_dim=12, n_hidden=2, mod_dim=5)
    model = TorchLevelSetWitness.build(cfg, seed=3)
    feats = torch.randn(48, 7)
    target = torch.randn(48, 5)
    code_before = model.code.detach().clone()
    sdf_before = model.out_sdf.weight.detach().clone()
    row = structured_sdf_prefit(
        model, feats, target, steps=30, lr=2e-3, subsample=48, seed=9
    )
    assert row["loss_final"] < row["loss_initial"]
    assert not torch.equal(model.out_sdf.weight, sdf_before)
    assert torch.equal(model.code, code_before)
    with pytest.raises(ValueError, match="pair-0/shared"):
        structured_sdf_prefit(
            model, feats[None], target, steps=1, lr=1e-3, subsample=8, seed=1
        )


def test_accum_pair_cursor_roundtrip_preserves_exact_coverage_order():
    cur = DeterministicPairCursor(n_pairs=5)
    assert cur.next_indices(3) == [0, 1, 2]
    cur.record_accepted(2)
    state = cur.state_dict()
    restored = DeterministicPairCursor(n_pairs=5)
    restored.load_state_dict(state)
    assert restored.next_indices(4) == [3, 4, 0, 1]
    assert restored.accepted_total == 2
    assert restored.attempted_total == 7


def test_epoch_pair_cursor_is_permuted_exhaustive_and_resumes_mid_epoch():
    cur = DeterministicPairCursor(n_pairs=11, seed=73)
    cur.begin_epoch(4)
    first = cur.next_epoch_indices(4)
    state = cur.state_dict()
    restored = DeterministicPairCursor(n_pairs=11, seed=999)
    restored.load_state_dict(state)
    rest = []
    while not restored.epoch_complete():
        rest.extend(restored.next_epoch_indices(4))
    assert len(first + rest) == 11
    assert sorted(first + rest) == list(range(11))
    assert restored.epoch == 4 and restored.epoch_complete()
    restored.begin_epoch(5)
    next_order = restored.next_epoch_indices(11)
    assert sorted(next_order) == list(range(11))
    assert next_order != first + rest


def test_per_group_clip_is_distinct_and_returns_device_tensors():
    cfg = CudaLevelSetConfig(n_pairs=1, in_feat=4, hidden_dim=6, n_hidden=1, mod_dim=3)
    model = TorchLevelSetWitness.build(cfg, seed=4)
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 10.0
    groups = parameter_groups(model)
    assert groups["muon"] and groups["adam"] and groups["code"]
    norms = clip_grad_groups(groups, 0.25)
    assert all(v is None or isinstance(v, torch.Tensor) for v in norms.values())
    for params in groups.values():
        grads = [p.grad.reshape(-1) for p in params if p.grad is not None]
        if grads:
            assert torch.linalg.vector_norm(torch.cat(grads)) <= 0.25001


def test_cpu_policy_and_cuda_graph_guard_fallbacks_are_truthful():
    policy = select_torch_execution_policy("cpu")
    assert policy.execution_label == "eager_fallback"
    assert policy.amp_dtype is None and not policy.cuda_graphs
    guard = CudaGraphRecaptureGuard()
    guard.mark_captured()
    guard.mark_replayed()  # ordinary weight updates do not invalidate capture
    guard.invalidate_control_layout()
    assert not guard.may_replay()
    with pytest.raises(RuntimeError, match="stale CUDA graph"):
        guard.mark_replayed()
    guard.mark_captured()
    guard.mark_replayed()


def test_cuda_policy_explicitly_disables_deterministic_algorithms(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", calls.append)
    policy = TorchExecutionPolicy(
        device_type="cuda", amp_dtype="bfloat16", grad_scaler=False,
        tf32=True, cudnn_benchmark=True, compile_mode="max-autotune",
        cuda_graphs=True, execution_label="megakernel_candidate",
    )
    apply_torch_execution_policy(policy)
    assert calls == [False]
    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32
    assert torch.backends.cudnn.benchmark
    assert not torch.backends.cudnn.deterministic


def test_compile_adoption_uses_functional_gate_not_gradient_identity(monkeypatch):
    cfg = CudaLevelSetConfig(n_pairs=1, in_feat=4, hidden_dim=6, n_hidden=1, mod_dim=3)
    model = TorchLevelSetWitness.build(cfg, seed=8)
    feats = torch.randn(11, 4)
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    row = compile_identity_probe(
        model, feats, torch.tensor([0]),
        lambda rgb, phi: rgb.square().mean() + phi.square().mean(),
    )
    assert row["argmax_equal"] and row["cosine_phi"] >= 0.9997
    assert row["adoptable"]
    assert row["training_loop_bit_identity_waiver"] is True
    assert "grad_max_abs_delta" in row


def test_pose_carrier_forward_survives_bf16_autocast_inputs():
    """$0 guard for the 2026-07-15 r4 H100 rc=1 (@415.4s, ~$0.58):
    torch.linalg.inv has no BFloat16 kernel, and under bf16 autocast the
    homography chain fed it a bf16 H. The warp must run fully in fp32 with
    autocast disabled (matching the fp32 NumPy/MLX reference warp) regardless
    of the surrounding autocast context or input dtype."""
    from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom

    geom = GroundHomographyGeom.eon(native_hw=(12, 16), pitch=-0.01)
    xi = np.array([[0.002, -0.001, 0.003, 0.0004, -0.0002, 0.0003]], np.float32)
    carrier = TorchPoseCarrier.build(xi, geom)
    src32 = torch.linspace(0.0, 255.0, 12 * 16 * 3).reshape(1, 12, 16, 3)
    baseline = carrier(src32, torch.tensor([0]))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        under_autocast = carrier(src32, torch.tensor([0]))
    assert under_autocast.dtype == torch.float32
    assert torch.allclose(under_autocast, baseline, atol=1e-6)
    src_bf16 = src32.to(torch.bfloat16)
    from_bf16_input = carrier(src_bf16, torch.tensor([0]))
    assert from_bf16_input.dtype == torch.float32
    assert torch.isfinite(from_bf16_input).all()
