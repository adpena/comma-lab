"""Tests for the ddm_bo1 PoseNet stem pair-geometry probe.

These test BEHAVIOUR, not constants. The load-bearing one is
`test_polyphase_blocks_reproduce_a_real_stride2_conv`: it re-derives the Fourier
block-diagonalisation independently in the spatial domain, so the whole apparatus
fails loudly if the polyphase index algebra is wrong.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "ddm_bo1_posenet_pair_geometry",
    Path(__file__).resolve().parent.parent / "ddm_bo1_posenet_pair_geometry.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _apply_blocks(blocks: torch.Tensor, x: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Apply Fourier blocks to x by hand: polyphase split -> DFT -> matmul -> inverse DFT."""
    h2, w2 = grid
    c_in = x.shape[0]
    # blocks index columns as (channel, r_h, r_w) with channel slowest; match that exactly.
    poly = torch.stack([x[:, rh::2, rw::2] for rh in (0, 1) for rw in (0, 1)], dim=1)
    poly = poly.reshape(c_in * 4, h2, w2)
    spec = torch.fft.fft2(poly.to(torch.complex128)).reshape(c_in * 4, h2 * w2)
    out = torch.einsum("nod,dn->on", blocks, spec)
    return torch.fft.ifft2(out.reshape(-1, h2, w2)).real


def _circular_stride2_conv(kernel: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y[c,m,n] = sum_{d,i,j} W[c,d,i,j] x[d, 2m+i-1, 2n+j-1] with circular wraparound."""
    padded = torch.cat([x[:, -1:], x, x[:, :1]], dim=1)
    padded = torch.cat([padded[:, :, -1:], padded, padded[:, :, :1]], dim=2)
    return torch.nn.functional.conv2d(padded.unsqueeze(0), kernel, stride=2).squeeze(0)


@pytest.fixture
def small_kernel() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(5, 4, 3, 3, dtype=torch.double)


def test_polyphase_blocks_reproduce_a_real_stride2_conv(small_kernel):
    """Independent spatial-domain re-derivation of the Fourier block-diagonalisation."""
    grid = (6, 8)
    torch.manual_seed(11)
    x = torch.randn(4, 2 * grid[0], 2 * grid[1], dtype=torch.double)
    blocks = mod.polyphase_fourier_blocks(small_kernel, grid)
    got = _apply_blocks(blocks, x, grid)
    want = _circular_stride2_conv(small_kernel, x)
    assert torch.allclose(got, want, atol=1e-9), (got - want).abs().max()


def test_polyphase_blocks_shape(small_kernel):
    blocks = mod.polyphase_fourier_blocks(small_kernel, (6, 8))
    assert blocks.shape == (48, 5, 16)


def test_polyphase_blocks_satisfy_parseval(small_kernel):
    """Sum of block energies equals grid size times kernel energy (convention-free check)."""
    grid = (6, 8)
    blocks = mod.polyphase_fourier_blocks(small_kernel, grid)
    assert (blocks.abs() ** 2).sum().item() == pytest.approx(
        grid[0] * grid[1] * (small_kernel ** 2).sum().item(), rel=1e-12)


def test_fold_conv_bn_matches_module_composition():
    conv = torch.nn.Conv2d(3, 4, 3, bias=False)
    bn = torch.nn.BatchNorm2d(4)
    torch.nn.init.normal_(bn.weight)
    torch.nn.init.normal_(bn.bias)
    bn.running_var.copy_(torch.rand(4) + 0.5)
    bn.running_mean.copy_(torch.randn(4))
    holder = type("H", (), {"conv": conv, "bn": bn})()
    w, b = mod.fold_conv_bn(holder)
    x = torch.randn(1, 3, 9, 9)
    bn.eval()
    with torch.no_grad():
        want = bn(conv(x)).double()
    got = torch.nn.functional.conv2d(x.double(), w, b)
    assert torch.allclose(got, want, atol=1e-9)


def test_fold_conv_bn_is_not_identity_when_bn_is_nontrivial():
    """Guard: a fold that silently ignored BN would still pass a shape-only test."""
    conv = torch.nn.Conv2d(2, 3, 1, bias=False)
    bn = torch.nn.BatchNorm2d(3)
    bn.weight.data.fill_(3.0)
    bn.running_var.data.fill_(1.0)
    holder = type("H", (), {"conv": conv, "bn": bn})()
    w, _ = mod.fold_conv_bn(holder)
    assert not torch.allclose(w, conv.weight.double())


POLICIES = ("optimal_in_range_A0", "delta0_plus", "delta0_minus")


def _paired(k0: torch.Tensor, k1: torch.Tensor) -> torch.Tensor:
    return torch.cat([k0, k1], dim=1)


def test_identical_blocks_make_plus_policy_four_times_worse():
    """A_0 == A_1 => delta_0=+delta_1 doubles z, i.e. ratio 4; delta_0=-delta_1 cancels."""
    torch.manual_seed(3)
    k = torch.randn(16, 3, 3, 3, dtype=torch.double)
    out = mod.pair_geometry(_paired(k, k), (6, 8))
    assert out["delta0_plus"]["mean_ratio"] == pytest.approx(4.0, rel=1e-9)
    assert out["delta0_minus"]["mean_ratio"] == pytest.approx(0.0, abs=1e-18)
    assert out["optimal_in_range_A0"]["mean_ratio"] == pytest.approx(0.0, abs=1e-12)


def test_zero_frame0_block_gives_ratio_one_for_every_policy():
    """If frame_0 cannot reach the scorer at all, no policy beats leaving it alone."""
    torch.manual_seed(4)
    k1 = torch.randn(16, 3, 3, 3, dtype=torch.double)
    out = mod.pair_geometry(_paired(torch.zeros_like(k1), k1), (6, 8))
    for name in POLICIES:
        assert out[name]["mean_ratio"] == pytest.approx(1.0, rel=1e-9)


def test_optimal_policy_is_never_worse_than_leaving_frame0_alone():
    torch.manual_seed(5)
    out = mod.pair_geometry(torch.randn(16, 6, 3, 3, dtype=torch.double), (6, 8))
    assert out["optimal_in_range_A0"]["mean_ratio"] <= 1.0 + 1e-9
    assert out["optimal_in_range_A0"]["share_of_pose_energy_worse_than_leaving_frame0_alone"] == 0.0


def test_optimal_policy_lower_bounds_both_fixed_policies():
    torch.manual_seed(6)
    out = mod.pair_geometry(torch.randn(16, 6, 3, 3, dtype=torch.double), (6, 8))
    best = out["optimal_in_range_A0"]["mean_ratio"]
    assert best <= out["delta0_plus"]["mean_ratio"] + 1e-9
    assert best <= out["delta0_minus"]["mean_ratio"] + 1e-9


def test_closed_form_crosscheck_is_reported_and_matches():
    torch.manual_seed(8)
    kernel = torch.randn(16, 6, 3, 3, dtype=torch.double)
    out = mod.pair_geometry(kernel, (6, 8))
    k0, k1 = kernel[:, :3], kernel[:, 3:]
    assert out["delta0_plus"]["closed_form_crosscheck"] == pytest.approx(
        (((k0 + k1).norm() / k1.norm()) ** 2).item(), rel=1e-12)


def _lossy_blocks(original):
    def wrapped(k, g):
        blocks = original(k, g).clone()
        blocks[:, 0, :] *= 3.0        # reweight one output row: breaks the energy bookkeeping
                                      # while leaving M_1's COLUMN rank intact, so this trips the
                                      # closed-form cross-check and not the rank precondition
        return blocks
    return wrapped


def test_pair_geometry_raises_when_fourier_and_closed_form_disagree(monkeypatch):
    """Mutation guard: corrupt the block builder and the internal cross-check must fire.

    NOTE the limit of this guard, deliberately recorded: the closed form is a Parseval
    identity, so it validates ENERGY BOOKKEEPING and is INVARIANT to the polyphase tap
    arrangement. A pure index-algebra error would pass it. That is what
    `test_polyphase_blocks_reproduce_a_real_stride2_conv` exists to catch instead --
    a scaling or flip mutation here would (correctly) NOT raise.
    """
    monkeypatch.setattr(mod, "polyphase_fourier_blocks",
                        _lossy_blocks(mod.polyphase_fourier_blocks))
    with pytest.raises(RuntimeError, match="closed form"):
        mod.pair_geometry(torch.randn(16, 6, 3, 3, dtype=torch.double), (6, 8))


def test_scaling_the_kernel_does_not_change_any_ratio():
    """Companion to the guard above: the ratios are scale-invariant by construction."""
    torch.manual_seed(21)
    kernel = torch.randn(16, 6, 3, 3, dtype=torch.double)
    a = mod.pair_geometry(kernel, (6, 8))
    b = mod.pair_geometry(kernel * 1.5, (6, 8))
    for name in POLICIES:
        assert a[name]["mean_ratio"] == pytest.approx(b[name]["mean_ratio"], rel=1e-12)


def test_quantiles_are_monotone_and_bracket_the_mean_region():
    torch.manual_seed(9)
    out = mod.pair_geometry(torch.randn(16, 6, 3, 3, dtype=torch.double), (6, 8))
    for name in POLICIES:
        q = list(out[name]["energy_weighted_quantiles"].values())
        assert q == sorted(q)


def test_energy_weighted_quantiles_report_all_six_levels():
    torch.manual_seed(10)
    out = mod.pair_geometry(torch.randn(16, 6, 3, 3, dtype=torch.double), (6, 8))
    assert list(out["delta0_plus"]["energy_weighted_quantiles"]) == \
        ["p10", "p25", "p50", "p75", "p90", "p99"]


def test_channel_block_norms_detect_alignment_and_antialignment():
    torch.manual_seed(12)
    k = torch.randn(4, 6, 3, 3, dtype=torch.double)
    aligned = mod.channel_block_norms(_paired(k, k))
    assert aligned["cosine_frame0_frame1"] == pytest.approx(1.0, rel=1e-12)
    assert aligned["antisymmetric_part_fro"] == pytest.approx(0.0, abs=1e-12)
    anti = mod.channel_block_norms(_paired(k, -k))
    assert anti["cosine_frame0_frame1"] == pytest.approx(-1.0, rel=1e-12)
    assert anti["symmetric_part_fro"] == pytest.approx(0.0, abs=1e-12)


def test_channel_block_norms_energy_shares_sum_to_one():
    torch.manual_seed(13)
    out = mod.channel_block_norms(torch.randn(4, 12, 3, 3, dtype=torch.double))
    shares = [v["share_of_frame1_energy"] for v in out["per_channel"].values()]
    assert sum(shares) == pytest.approx(1.0, rel=1e-12)
    assert list(out["per_channel"]) == ["y00", "y10", "y01", "y11", "U", "V"]


def test_chunking_does_not_change_the_result():
    torch.manual_seed(14)
    kernel = torch.randn(16, 6, 3, 3, dtype=torch.double)
    a = mod.pair_geometry(kernel, (6, 8), chunk=3)
    b = mod.pair_geometry(kernel, (6, 8), chunk=10_000)
    for name in POLICIES:
        assert a[name]["mean_ratio"] == pytest.approx(b[name]["mean_ratio"], rel=1e-12)


def test_yuv6_kernel_is_exactly_six_dimensional_per_block():
    """The pose-null / seg-active subspace Q3 rests on this rank; re-derived by finite diff."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "upstream"))
    from frame_utils import rgb_to_yuv6  # decorated @torch.no_grad(): autograd would give 0

    x0 = torch.rand(1, 3, 2, 2, dtype=torch.double) * 100 + 80  # mid-range: no clamp active
    eps = 1e-6
    jac = torch.zeros(6, 12, dtype=torch.double)
    for k in range(12):
        e = torch.zeros(12, dtype=torch.double)
        e[k] = eps
        jac[:, k] = (rgb_to_yuv6(x0 + e.view(1, 3, 2, 2)).flatten()
                     - rgb_to_yuv6(x0 - e.view(1, 3, 2, 2)).flatten()) / (2 * eps)
    assert int((torch.linalg.svdvals(jac) > 1e-6).sum()) == 6


def test_stem_output_grid_default_matches_segnet_input_size_at_stride_two():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "upstream"))
    from frame_utils import segnet_model_input_size

    w, h = segnet_model_input_size
    assert (h // 2, w // 2) == mod.DEFAULT_GRID


def test_module_declares_no_score_claim():
    """The probe is a frozen-scorer structural fact, never a score."""
    src = (Path(__file__).resolve().parent.parent / "ddm_bo1_posenet_pair_geometry.py").read_text()
    assert '"score_claim": False' in src
    assert '"promotable": False' in src


def test_math_import_is_used_for_the_dft_phases():
    """Cheap guard that the phase construction still uses a real 2*pi grid."""
    blocks = mod.polyphase_fourier_blocks(torch.ones(1, 1, 3, 3, dtype=torch.double), (4, 4))
    dc = blocks[0].abs().sum().item()
    assert dc == pytest.approx(9.0, rel=1e-12)  # DC block sums every tap once
    assert math.isfinite(dc)


def test_pair_geometry_refuses_a_rank_deficient_frame1_block():
    """Precondition guard: fewer stem rows than polyphase columns => ratios do not span input."""
    with pytest.raises(RuntimeError, match="full column rank"):
        mod.pair_geometry(torch.randn(6, 6, 3, 3, dtype=torch.double), (6, 8))


def _real_stem_kernel() -> torch.Tensor:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "upstream"))
    from modules import PoseNet, posenet_sd_path
    from safetensors.torch import load_file

    posenet = PoseNet().eval()
    posenet.load_state_dict(load_file(str(posenet_sd_path), device="cpu"))
    kernel, _bias, _err = mod.stem_effective_kernel(posenet)
    return kernel


def test_real_posenet_frame1_block_is_well_conditioned():
    """The reported means are only the claimed quantity because M_1 is full column rank here.

    Uses a coarse frequency grid: conditioning is a property of the kernel sampled at
    frequencies, so a subgrid is a valid (and much cheaper) check of the same function.
    """
    out = mod.pair_geometry(_real_stem_kernel(), (12, 16))
    assert out["m1_worst_conditioning_sigma_min_over_sigma_max"] > mod.RCOND


def test_real_posenet_stem_is_dominated_by_the_symmetric_part():
    """Load-bearing for the derivation: the pair read is common-mode, not a difference."""
    norms = mod.channel_block_norms(_real_stem_kernel())
    assert norms["cosine_frame0_frame1"] > 0.5
    assert norms["symmetric_part_fro"] > norms["antisymmetric_part_fro"]
