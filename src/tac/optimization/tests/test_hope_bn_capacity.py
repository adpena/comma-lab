# SPDX-License-Identifier: MIT
"""Tests for the HOPE BN-capacity generator (task #725, arm hb1).

Every test is synthetic and self-contained: no contest artifacts, no SSD
paths, no network. The behaviours under test are the ones the sealed RG3
parity + capacity receipts depend on.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.optimization.hope_bn_capacity import (
    HEAD_IN_CHANNELS,
    KernelAccumulator,
    enumerate_segnet_units,
    fisher_fine_band,
    fisher_trace_map,
    head_class_pair_delta_norms,
    head_difference_rank,
    load_bucket_index,
    relu_family_check,
    relu_gaussian_self_kernel,
    select_fine_band,
    site_local_capacity_field,
)

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Closed-form ReLU surrogate kernel (HOPE Eq. 79)
# ---------------------------------------------------------------------------


def test_relu_gaussian_self_kernel_matches_monte_carlo() -> None:
    rng = np.random.default_rng(7)
    gamma = np.array([0.5, 1.0, 2.0, 0.3], dtype=np.float64)
    beta = np.array([-1.0, 0.0, 0.7, 2.5], dtype=np.float64)
    k = relu_gaussian_self_kernel(gamma, beta)
    n = 2_000_000
    y = beta[None, :] + gamma[None, :] * rng.standard_normal((n, 4))
    mc = np.mean(np.maximum(y, 0.0) ** 2, axis=0)
    # MC noise floor at 2e6 samples: loosest on the tiny K (beta=-1, gamma=0.5)
    np.testing.assert_allclose(k, mc, rtol=2.5e-2, atol=1e-5)


def test_relu_gaussian_self_kernel_degenerate_gamma() -> None:
    k = relu_gaussian_self_kernel(np.array([0.0, 0.0]), np.array([2.0, -2.0]))
    np.testing.assert_allclose(k, [4.0, 0.0])


def test_relu_gaussian_self_kernel_shape_mismatch_refuses() -> None:
    with pytest.raises(DirectDescriptionError):
        relu_gaussian_self_kernel(np.zeros(3), np.zeros(2))


# ---------------------------------------------------------------------------
# Fisher trace + fine-band selection (parity semantics with the sealed RG3
# hand derivation: clip 40, 0.5*sech^2(m/2), fp64 subband sums, earliest-max)
# ---------------------------------------------------------------------------


def test_fisher_trace_matches_reference_formula() -> None:
    m = np.abs(np.random.default_rng(0).standard_normal((384, 512))).astype(np.float32) * 3.0
    trace = fisher_trace_map(m)
    clipped = np.minimum(m, np.float32(40.0))
    ref = np.float32(0.5) / np.cosh(clipped * np.float32(0.5)) ** np.float32(2.0)
    np.testing.assert_array_equal(trace, ref)


def test_fisher_trace_refuses_negative_or_nonfinite() -> None:
    bad = np.full((384, 512), -1.0, dtype=np.float32)
    with pytest.raises(DirectDescriptionError):
        fisher_trace_map(bad)
    nan = np.full((384, 512), np.nan, dtype=np.float32)
    with pytest.raises(DirectDescriptionError):
        fisher_trace_map(nan)


def test_select_fine_band_picks_max_and_earliest_on_tie() -> None:
    mass = np.zeros((384, 512), dtype=np.float64)
    # row band 2 spans rows 128..191; fine bands are 16 rows each
    mass[128 + 16 * 3] = 1.0  # fine 3
    assert select_fine_band(mass, row_band=2) == 3
    mass[128 + 16 * 1] = 1.0  # tie between fine 1 and fine 3 -> earliest
    assert select_fine_band(mass, row_band=2) == 1


def test_select_fine_band_refuses_empty_support() -> None:
    with pytest.raises(DirectDescriptionError):
        select_fine_band(np.zeros((384, 512)), row_band=0)


def test_fisher_fine_band_parity_vs_receiver_grammar_reference() -> None:
    """My selector must equal derive_rg3_fisher_margin_band on shared inputs."""

    from types import SimpleNamespace

    from tac.optimization.ddm_rg1_receiver_grammar import derive_rg3_fisher_margin_band

    rng = np.random.default_rng(11)
    margin = np.abs(rng.standard_normal((384, 512))).astype(np.float32) * 4.0
    mask_a = rng.random((384, 512)) < 0.20
    mask_b = rng.random((384, 512)) < 0.15

    class _Base:
        predictor = SimpleNamespace(source_pair_start=0)
        z = SimpleNamespace(n_pairs=600)
        layers = (SimpleNamespace(role="Road"), SimpleNamespace(role="Movable"))

        def _mask_for_layer(self, layer, local_pair_id, replace_g1_movable=True):
            return mask_a if layer.role == "Road" else mask_b

    for row_band in range(6):
        support = mask_a | mask_b
        if not support[row_band * 64 : (row_band + 1) * 64].any():
            continue
        ref = derive_rg3_fisher_margin_band(
            _Base(), pair_index=0, class_a=0, class_b=3, row_band=row_band, margin_map=margin
        )
        mine = fisher_fine_band(margin, support, row_band=row_band)
        assert mine == ref


def test_fisher_fine_band_flat_site_weight_is_parity() -> None:
    rng = np.random.default_rng(3)
    margin = np.abs(rng.standard_normal((384, 512))).astype(np.float32)
    support = rng.random((384, 512)) < 0.3
    flat = np.full((384, 512), 7.5)
    assert fisher_fine_band(margin, support, row_band=1) == fisher_fine_band(
        margin, support, row_band=1, site_weight=flat
    )


def test_fisher_fine_band_site_weight_can_move_selection() -> None:
    margin = np.ones((384, 512), dtype=np.float32)
    support = np.zeros((384, 512), dtype=bool)
    support[0:16] = True  # fine 0 of row band 0
    support[16:32] = True  # fine 1
    # parity: equal trace, equal support -> earliest (fine 0)
    assert fisher_fine_band(margin, support, row_band=0) == 0
    weight = np.ones((384, 512))
    weight[16:32] = 10.0
    assert fisher_fine_band(margin, support, row_band=0, site_weight=weight) == 1


def test_fisher_fine_band_refuses_bad_site_weight() -> None:
    margin = np.ones((384, 512), dtype=np.float32)
    support = np.ones((384, 512), dtype=bool)
    with pytest.raises(DirectDescriptionError):
        fisher_fine_band(margin, support, row_band=0, site_weight=-np.ones((384, 512)))


# ---------------------------------------------------------------------------
# Kernel accumulator + gauge invariance of the capacity on a tiny BN net
# ---------------------------------------------------------------------------


def _empirical_k(module: torch.nn.Module, hook_on: torch.nn.Module, x: torch.Tensor, n_ch: int) -> np.ndarray:
    acc = KernelAccumulator(n_ch)
    h = hook_on.register_forward_hook(lambda _m, _i, out: acc.update(out))
    try:
        with torch.inference_mode():
            module(x)
    finally:
        h.remove()
    return acc.k_diag()


def test_kernel_accumulator_second_moment() -> None:
    acc = KernelAccumulator(2)
    t = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])  # (1, 2, 1, 2)
    acc.update(t)
    np.testing.assert_allclose(acc.k_diag(), [(1 + 4) / 2, (9 + 16) / 2])
    np.testing.assert_allclose(acc.mean(), [1.5, 3.5])


def test_capacity_gauge_invariance_bn_relu_unit() -> None:
    """HOPE's scale-symmetry quotient: (gamma,beta)*lam with w_out/lam keeps
    ||w_out||*sqrt(K) invariant for a PH-1 (ReLU) unit."""

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1, bias=False),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 4, 3, padding=1, bias=False),
    ).eval()
    with torch.no_grad():
        net[1].weight.uniform_(0.5, 1.5)
        net[1].bias.uniform_(-0.5, 0.5)
        net[1].running_mean.uniform_(-0.2, 0.2)
        net[1].running_var.uniform_(0.5, 1.5)
    x = torch.randn(4, 3, 16, 16)

    def capacity() -> np.ndarray:
        k = _empirical_k(net, net[2], x, 8)
        w = net[3].weight.detach().numpy().astype(np.float64)
        w_out = np.sqrt((w**2).sum(axis=(0, 2, 3)))
        return w_out * np.sqrt(k)

    cap0 = capacity()
    lam = 3.7
    with torch.no_grad():
        net[1].weight.mul_(lam)
        net[1].bias.mul_(lam)
        net[3].weight.div_(lam)
    cap1 = capacity()
    np.testing.assert_allclose(cap0, cap1, rtol=1e-5)


# ---------------------------------------------------------------------------
# Unit enumeration + consumer resolution on the real architecture (random
# weights: architecture-only, no frozen checkpoint required)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def random_segnet():
    smp = pytest.importorskip("segmentation_models_pytorch")
    seg = smp.Unet("tu-efficientnet_b2", classes=5, activation=None, encoder_weights=None)
    return seg.eval()


def test_enumerate_units_counts_and_families(random_segnet) -> None:
    units = enumerate_segnet_units(random_segnet)
    assert len(units) == 78  # 68 encoder BNAct + 10 decoder BN+ReLU
    dec = [u for u in units if u.stage == "decoder"]
    assert len(dec) == 10
    assert all(u.activation == "relu" for u in dec)
    assert all(u.consumer_status == "RESOLVED" for u in dec)
    enc = [u for u in units if u.stage == "encoder"]
    assert {u.activation for u in enc} <= {"silu", "identity"}
    check = relu_family_check(units)
    assert check["scorer_is_pure_relu_family"] is False  # the charter caveat, measured


def test_final_decoder_unit_feeds_head(random_segnet) -> None:
    units = enumerate_segnet_units(random_segnet)
    head_units = [u for u in units if u.consumer_paths == ("segmentation_head.0",)]
    assert len(head_units) == 1
    assert head_units[0].n_channels == HEAD_IN_CHANNELS


def test_consumer_weight_norms_decoder_chain(random_segnet) -> None:
    from tac.optimization.hope_bn_capacity import consumer_weight_norms

    units = {u.unit_id: u for u in enumerate_segnet_units(random_segnet)}
    # block0.conv1 -> consumed fully by block0.conv2
    u = units["dec.blocks.0.conv1"]
    norms = consumer_weight_norms(random_segnet, u)
    w = random_segnet.decoder.blocks[0].conv2[0].weight.detach().numpy().astype(np.float64)
    np.testing.assert_allclose(norms, np.sqrt((w**2).sum(axis=(0, 2, 3))), rtol=1e-6)
    # block0.conv2 -> first 256 input channels of block1.conv1
    u2 = units["dec.blocks.0.conv2"]
    norms2 = consumer_weight_norms(random_segnet, u2)
    w2 = random_segnet.decoder.blocks[1].conv1[0].weight.detach().numpy().astype(np.float64)
    np.testing.assert_allclose(norms2, np.sqrt((w2[:, :256] ** 2).sum(axis=(0, 2, 3))), rtol=1e-6)


def test_consumer_weight_norms_unresolved_returns_none(random_segnet) -> None:
    from tac.optimization.hope_bn_capacity import consumer_weight_norms

    units = enumerate_segnet_units(random_segnet)
    unresolved = [u for u in units if u.consumer_status == "UNRESOLVED_CONSUMER_GRAPH_V1"]
    assert unresolved, "expected some honestly-unresolved encoder units"
    assert consumer_weight_norms(random_segnet, unresolved[0]) is None


def test_depthwise_consumer_norms(random_segnet) -> None:
    from tac.optimization.hope_bn_capacity import consumer_weight_norms

    units = enumerate_segnet_units(random_segnet)
    dw_units = [
        u
        for u in units
        if u.consumer_status == "RESOLVED" and u.stage == "encoder" and u.consumer_paths and "conv_dw" in u.consumer_paths[0]
    ]
    assert dw_units, "expected InvertedResidual bn1 -> conv_dw units"
    u = dw_units[0]
    conv = dict(random_segnet.named_modules())[u.consumer_paths[0]]
    assert conv.groups == conv.in_channels  # honest depthwise
    norms = consumer_weight_norms(random_segnet, u)
    w = conv.weight.detach().numpy().astype(np.float64)
    np.testing.assert_allclose(norms, np.sqrt((w[:, 0] ** 2).sum(axis=(1, 2))), rtol=1e-6)


# ---------------------------------------------------------------------------
# Rank-4 head composition
# ---------------------------------------------------------------------------


def test_head_class_pair_delta_norms_and_rank() -> None:
    rng = np.random.default_rng(5)
    w = rng.standard_normal((5, 16, 3, 3))
    norms = head_class_pair_delta_norms(w)
    assert set(norms) == {f"{a}-{b}" for a in range(5) for b in range(a + 1, 5)}
    d01 = w[0] - w[1]
    np.testing.assert_allclose(norms["0-1"], np.sqrt((d01**2).sum(axis=(1, 2))))
    assert head_difference_rank(w)["rank"] == 4
    # a rank-deficient head: all classes share one direction
    w_low = np.broadcast_to(w[0], (5, 16, 3, 3)).copy()
    w_low[1] = 2.0 * w[0]
    assert head_difference_rank(w_low)["rank"] == 1


def test_head_delta_norms_shape_refusal() -> None:
    with pytest.raises(DirectDescriptionError):
        head_class_pair_delta_norms(np.zeros((4, 16, 3, 3)))


# ---------------------------------------------------------------------------
# Site-local capacity field
# ---------------------------------------------------------------------------


def test_site_local_capacity_field_math() -> None:
    f = np.zeros((16, 2, 2))
    f[0] = [[1.0, 2.0], [0.0, 1.0]]
    f[3] = [[2.0, 0.0], [1.0, 1.0]]
    cap = np.zeros(16)
    cap[0], cap[3] = 1.0, 3.0
    field = site_local_capacity_field(f, cap)
    expect = 0.25 * f[0] ** 2 + 0.75 * f[3] ** 2
    np.testing.assert_allclose(field, expect)


def test_site_local_capacity_field_refusals() -> None:
    with pytest.raises(DirectDescriptionError):
        site_local_capacity_field(np.zeros((15, 2, 2)), np.ones(16))
    with pytest.raises(DirectDescriptionError):
        site_local_capacity_field(np.zeros((16, 2, 2)), np.zeros(16))
    with pytest.raises(DirectDescriptionError):
        site_local_capacity_field(np.zeros((16, 2, 2)), -np.ones(16))


# ---------------------------------------------------------------------------
# Bucket id -> canonical class pair parsing
# ---------------------------------------------------------------------------


def test_bucket_class_pair_canonical_order() -> None:
    from tac.optimization.hope_bn_capacity import bucket_class_pair

    assert bucket_class_pair("road_movable__cell__static_in_image") == (0, 3)
    assert bucket_class_pair("lane_undrivable__boundary__transient") == (1, 2)
    assert bucket_class_pair("undrivable_movable__boundary__static_in_xi_proxy") == (2, 3)
    assert bucket_class_pair("movable_mycar__cell__transient") == (3, 4)
    with pytest.raises(DirectDescriptionError):
        bucket_class_pair("sky_road__cell__static_in_image")
    with pytest.raises(DirectDescriptionError):
        bucket_class_pair("road_road__cell__static_in_image")


# ---------------------------------------------------------------------------
# Bucket index loading (synthetic fixture with its own custody sha)
# ---------------------------------------------------------------------------


def test_load_bucket_index_groups_per_pair(tmp_path) -> None:
    plane = 384 * 512
    flat = np.array(
        [3 * plane + 7, 3 * plane + 9, 5 * plane + 0, 0 * plane + plane - 1],
        dtype=np.uint32,
    )
    p = tmp_path / "idx.npz"
    np.savez(p, bucket_x=flat)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    out = load_bucket_index(p, {"my_bucket": "bucket_x"}, expected_sha256=sha)
    per_pair = out["my_bucket"]
    assert set(per_pair) == {0, 3, 5}
    np.testing.assert_array_equal(per_pair[3], [7, 9])
    np.testing.assert_array_equal(per_pair[0], [plane - 1])


def test_load_bucket_index_refuses_wrong_sha(tmp_path) -> None:
    p = tmp_path / "idx.npz"
    np.savez(p, bucket_x=np.array([0], dtype=np.uint32))
    with pytest.raises(DirectDescriptionError):
        load_bucket_index(p, {"b": "bucket_x"}, expected_sha256="0" * 64)


def test_load_bucket_index_refuses_out_of_grid(tmp_path) -> None:
    plane = 384 * 512
    p = tmp_path / "idx.npz"
    np.savez(p, bucket_x=np.array([600 * plane], dtype=np.uint32))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(DirectDescriptionError):
        load_bucket_index(p, {"b": "bucket_x"}, expected_sha256=sha)


# ---------------------------------------------------------------------------
# End-to-end miniature measurement (tiny random SegNet-shaped pass is too
# heavy for CI; instead verify the accumulator/hook contract on a stub net)
# ---------------------------------------------------------------------------


def test_measurement_contract_on_stub_unit() -> None:
    """Hook-on-activation-output measures E[psi^2] exactly (fp64)."""

    net = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1, bias=False), torch.nn.BatchNorm2d(3), torch.nn.ReLU()).eval()
    torch.manual_seed(1)
    x = torch.randn(8, 2, 6, 6)
    k = _empirical_k(net, net[2], x, 3)
    with torch.inference_mode():
        psi = net(x)
    ref = (psi.double() ** 2).mean(dim=(0, 2, 3)).numpy()
    # accumulator squares in fp32 before the fp64 sum; ref squares in fp64
    np.testing.assert_allclose(k, ref, rtol=1e-6)
    assert math.isfinite(float(k.sum()))
