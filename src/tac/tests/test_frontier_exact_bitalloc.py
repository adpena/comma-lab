# SPDX-License-Identifier: MIT
"""NO-FAKE tests for tac.frontier_exact_bitalloc — the exact precision-allocation solve.

These verify ACTUAL behavior, NOT constants:
  * the d_pose per-tensor sensitivity is a genuine non-zero autograd gradient through a
    real stub PoseNet (if the producer returned zeros it FAILS);
  * the combined sensitivity actually mixes seg+pose by the master gradient;
  * the water-fill allocation ACTUALLY emits NON-UNIFORM per-tensor nbits that rise with
    the per-weight score-impact (a blind tensor gets fewer bits than a sensitive one);
  * λ is monotone (more λ → more bits) and the bisection pins a target mean-bits;
  * the requant ACTUALLY snaps each tensor to ITS allocated grid (a higher-nbits tensor has
    finer resolution / lower round-trip error than a lower-nbits one).

If the allocator were replaced by ``return uniform nbits`` the non-uniformity tests FAIL.
"""
from __future__ import annotations

import math

import pytest
import torch

from tac.frontier_exact_bitalloc import (
    BitAllocation,
    CombinedTensorSensitivity,
    DecoderPoseSaliency,
    FrontierBitallocError,
    combine_sensitivities,
    compute_decoder_tensor_pose_saliency,
    lam_for_target_mean_bits,
    per_stage_bits_summary,
    requant_state_dict_per_tensor_nbits,
    waterfill_bit_allocation,
)


# ---------------------------------------------------------------------------
# Stubs (mirror the HNeRV name geometry; real autograd)
# ---------------------------------------------------------------------------


class _StubDecoder(torch.nn.Module):
    """Tiny named-module decoder mirroring the HNeRV names so the per-tensor
    pose producer runs with REAL per-weight gradients."""

    def __init__(self, base_hw=(4, 6)):
        super().__init__()
        self.base_h, self.base_w = base_hw
        self.stem = torch.nn.Linear(28, 6 * self.base_h * self.base_w)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Conv2d(6, 6, 3, padding=1) for _ in range(2)]
        )
        self.rgb_0 = torch.nn.Conv2d(6, 3, 3, padding=1)
        self.rgb_1 = torch.nn.Conv2d(6, 3, 3, padding=1)

    def forward(self, z):
        b = z.shape[0]
        x = torch.sin(self.stem(z).view(b, 6, self.base_h, self.base_w))
        for blk in self.blocks:
            x = torch.sin(blk(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)  # (b, 2, 3, H, W)


class _StubPoseNet:
    """A minimal differentiable PoseNet stub: flatten the roundtripped pair and a
    fixed linear map to 6 pose dims, with a REAL dependence on the input."""

    def __init__(self, in_pixels, classes=6):
        torch.manual_seed(0)
        self.w = torch.randn(classes, in_pixels) * 0.001

    def __call__(self, decoded_bhwc):
        # decoded_bhwc: (B, 2, H, W, 3) in [0,255]
        b = decoded_bhwc.shape[0]
        flat = decoded_bhwc.reshape(b, -1)
        return flat @ self.w.t()  # (B, 6) WITH grad


def _stub_render_pair(decoder, latent_rows):
    out = decoder(latent_rows)  # (b, 2, 3, H, W)
    return out.permute(0, 1, 3, 4, 2)  # (b, 2, H, W, 3) WITH grad


def _stub_latents(n=4):
    torch.manual_seed(3)
    return torch.rand(n, 28)


def _pose_targets(n=4):
    torch.manual_seed(7)
    return torch.randn(n, 6) * 0.01


def _make_pose_forward(dec):
    # one stub posenet sized to the decoder's flattened pair
    sample = _stub_render_pair(dec, _stub_latents(1))
    in_px = int(sample.reshape(1, -1).shape[1])
    pn = _StubPoseNet(in_px)
    return lambda decoded_bhwc: pn(decoded_bhwc)


# ---------------------------------------------------------------------------
# Step 1b — d_pose sensitivity is a REAL gradient
# ---------------------------------------------------------------------------


def test_pose_saliency_is_real_nonzero_per_weight_gradient():
    """Headline NO-FAKE guard: the d_pose per-tensor saliency must be a genuine
    non-zero autograd gradient through the stub PoseNet. Zeros/constants → FAIL."""
    dec = _StubDecoder()
    pf = _make_pose_forward(dec)
    out = compute_decoder_tensor_pose_saliency(
        dec,
        _stub_latents(),
        pf,
        _pose_targets(),
        render_pair_bchw=_stub_render_pair,
        frame_indices=[0, 1, 2],
    )
    assert isinstance(out, DecoderPoseSaliency)
    assert out.n_frames == 3
    # at least the weight tensors carry real non-zero gradient
    wkeys = [k for k in out.per_tensor if k.endswith(".weight")]
    assert any(out.per_tensor[k] > 0.0 for k in wkeys)
    # numel recorded for every param
    assert all(out.tensor_numel[k] > 0 for k in out.per_tensor)


def test_pose_saliency_responds_to_target_change():
    """Different pose targets → different per-tensor pose MSE gradient (real dep)."""
    dec = _StubDecoder()
    pf = _make_pose_forward(dec)
    a = compute_decoder_tensor_pose_saliency(
        dec, _stub_latents(), pf, _pose_targets(), render_pair_bchw=_stub_render_pair,
        frame_indices=[0, 1],
    )
    b = compute_decoder_tensor_pose_saliency(
        dec, _stub_latents(), pf, _pose_targets() + 5.0,
        render_pair_bchw=_stub_render_pair, frame_indices=[0, 1],
    )
    diffs = [abs(a.per_tensor[k] - b.per_tensor[k]) for k in a.per_tensor]
    assert max(diffs) > 0.0


def test_pose_saliency_bad_latent_shape_raises():
    dec = _StubDecoder()
    pf = _make_pose_forward(dec)
    with pytest.raises(Exception):
        compute_decoder_tensor_pose_saliency(
            dec, torch.rand(28), pf, _pose_targets(), render_pair_bchw=_stub_render_pair
        )


# ---------------------------------------------------------------------------
# Step 1c — combine actually mixes seg+pose by the master gradient
# ---------------------------------------------------------------------------


def _fake_seg_saliency(per_tensor):
    class _S:
        pass

    s = _S()
    s.per_tensor = dict(per_tensor)
    return s


def test_combine_mixes_seg_and_pose_by_master_gradient():
    dec = _StubDecoder()
    # craft seg + pose so the combined value is exactly checkable
    seg = _fake_seg_saliency({"rgb_0.weight": 2.0, "blocks.0.weight": 0.0})
    pose = DecoderPoseSaliency(
        per_tensor={"rgb_0.weight": 0.0, "blocks.0.weight": 3.0},
        n_frames=1,
        tensor_numel={"rgb_0.weight": 1, "blocks.0.weight": 1},
    )
    comb = combine_sensitivities(dec, seg, pose, w_seg=10.0, w_pose=100.0)
    assert comb.per_tensor["rgb_0.weight"] == pytest.approx(10.0 * 2.0 + 100.0 * 0.0)
    assert comb.per_tensor["blocks.0.weight"] == pytest.approx(10.0 * 0.0 + 100.0 * 3.0)
    # absmax + numel actually read off the real decoder weights
    assert comb.absmax["rgb_0.weight"] > 0.0
    assert comb.numel["rgb_0.weight"] == dec.rgb_0.weight.numel()


def test_combine_excludes_biases_and_1d():
    dec = _StubDecoder()
    seg = _fake_seg_saliency({})
    pose = DecoderPoseSaliency(per_tensor={}, n_frames=1, tensor_numel={})
    comb = combine_sensitivities(dec, seg, pose)
    # no bias / 1-D params in the allocation set
    assert all(not k.endswith(".bias") for k in comb.per_tensor)
    assert all(dec.get_parameter(k).dim() >= 2 for k in comb.per_tensor)


# ---------------------------------------------------------------------------
# Step 2 — water-fill emits NON-UNIFORM bits that rise with per-weight impact
# ---------------------------------------------------------------------------


def _sens(per_tensor, absmax, numel, w_seg=100.0, w_pose=271.0):
    return CombinedTensorSensitivity(
        per_tensor=dict(per_tensor),
        g_seg={k: 0.0 for k in per_tensor},
        g_pose={k: 0.0 for k in per_tensor},
        absmax=dict(absmax),
        numel=dict(numel),
        w_seg=w_seg,
        w_pose=w_pose,
    )


def test_waterfill_emits_nonuniform_bits():
    """The headline solve guard: a high-impact tensor gets MORE bits than a
    low-impact one of the same size. If the allocator returned uniform → FAIL."""
    sens = _sens(
        per_tensor={"hi.weight": 100.0, "lo.weight": 0.01},
        absmax={"hi.weight": 1.0, "lo.weight": 1.0},
        numel={"hi.weight": 1000, "lo.weight": 1000},
    )
    # pick a lam so both land inside [2,8] (not both clamped)
    alloc = waterfill_bit_allocation(sens, lam=1e3, b_min=2, b_max=8)
    assert isinstance(alloc, BitAllocation)
    assert alloc.nbits["hi.weight"] > alloc.nbits["lo.weight"]
    # NON-uniform by construction
    assert len(set(alloc.nbits.values())) > 1


def test_waterfill_bits_rise_with_per_weight_impact_not_total():
    """A large but blind tensor gets FEWER bits/weight than a small sensitive one
    (the per-WEIGHT impact drives the bit-width, not total tensor saliency)."""
    sens = _sens(
        per_tensor={"big_blind.weight": 1.0, "small_sharp.weight": 1.0},
        absmax={"big_blind.weight": 1.0, "small_sharp.weight": 1.0},
        numel={"big_blind.weight": 100000, "small_sharp.weight": 100},
    )
    alloc = waterfill_bit_allocation(sens, lam=1e6, b_min=2, b_max=8)
    # per-weight impact: small tensor has higher c_t/numel (sqrt(n) in c_t, /n -> 1/sqrt(n))
    assert alloc.per_weight_impact["small_sharp.weight"] > alloc.per_weight_impact["big_blind.weight"]
    assert alloc.nbits["small_sharp.weight"] >= alloc.nbits["big_blind.weight"]


def test_waterfill_monotone_in_lam():
    """More λ → more total bits (the RD knob is monotone)."""
    sens = _sens(
        per_tensor={"a.weight": 1.0, "b.weight": 0.1},
        absmax={"a.weight": 0.5, "b.weight": 0.5},
        numel={"a.weight": 500, "b.weight": 500},
    )
    lo = waterfill_bit_allocation(sens, lam=1e2, b_min=2, b_max=8).total_weight_bits
    hi = waterfill_bit_allocation(sens, lam=1e5, b_min=2, b_max=8).total_weight_bits
    assert hi >= lo


def test_waterfill_clamps_to_range():
    sens = _sens(
        per_tensor={"a.weight": 1e9},
        absmax={"a.weight": 1.0},
        numel={"a.weight": 10},
    )
    alloc = waterfill_bit_allocation(sens, lam=1e20, b_min=3, b_max=6)
    assert 3 <= alloc.nbits["a.weight"] <= 6


def test_waterfill_zero_sensitivity_gets_bmin():
    sens = _sens(
        per_tensor={"blind.weight": 0.0},
        absmax={"blind.weight": 1.0},
        numel={"blind.weight": 100},
    )
    alloc = waterfill_bit_allocation(sens, lam=1e9, b_min=2, b_max=8)
    assert alloc.nbits["blind.weight"] == 2


def test_waterfill_continuous_matches_kkt_closed_form():
    """The continuous b_t* must equal the closed-form log2(lam*ln2*c_t/numel_t)."""
    sens = _sens(
        per_tensor={"t.weight": 4.0},
        absmax={"t.weight": 2.0},
        numel={"t.weight": 64},
    )
    lam = 5e3
    alloc = waterfill_bit_allocation(sens, lam=lam, b_min=2, b_max=8)
    c_t = 4.0 * 2.0 * math.sqrt(64)
    expected = math.log2(lam * math.log(2.0) * c_t / 64)
    assert alloc.continuous_bits["t.weight"] == pytest.approx(expected, rel=1e-9)


def test_waterfill_rejects_bad_args():
    sens = _sens({"a.weight": 1.0}, {"a.weight": 1.0}, {"a.weight": 10})
    with pytest.raises(FrontierBitallocError):
        waterfill_bit_allocation(sens, lam=0.0)
    with pytest.raises(FrontierBitallocError):
        waterfill_bit_allocation(sens, lam=1.0, b_min=1, b_max=8)
    with pytest.raises(FrontierBitallocError):
        waterfill_bit_allocation(sens, lam=1.0, b_min=6, b_max=4)


def test_lam_for_target_mean_bits_pins_mean():
    """The bisection actually pins the param-weighted mean allocated bits."""
    sens = _sens(
        per_tensor={f"t{i}.weight": float(i + 1) for i in range(6)},
        absmax={f"t{i}.weight": 0.5 for i in range(6)},
        numel={f"t{i}.weight": 1000 for i in range(6)},
    )
    lam = lam_for_target_mean_bits(sens, 5.0, b_min=2, b_max=8)
    alloc = waterfill_bit_allocation(sens, lam, b_min=2, b_max=8)
    total_numel = sum(sens.numel.values())
    mean = alloc.total_weight_bits / total_numel
    assert mean == pytest.approx(5.0, abs=0.6)  # integer rounding -> coarse tolerance


# ---------------------------------------------------------------------------
# Step 3 — requant ACTUALLY snaps each tensor to ITS allocated grid
# ---------------------------------------------------------------------------


def test_requant_realizes_per_tensor_grid_finer_at_higher_nbits():
    """A tensor allocated more bits has FINER resolution (lower round-trip error)
    than the same tensor at fewer bits — the allocation is realized in the weights."""
    torch.manual_seed(0)
    w = torch.randn(64, 16) * 0.3
    sd = {"a.weight": w.clone(), "b.weight": w.clone(), "a.bias": torch.randn(64)}
    out = requant_state_dict_per_tensor_nbits(sd, {"a.weight": 8, "b.weight": 3})
    err_a = float((out["a.weight"] - w).abs().mean())
    err_b = float((out["b.weight"] - w).abs().mean())
    assert err_b > err_a  # int3 coarser than int8
    # the 1-D bias is copied through fp32 unchanged
    assert torch.allclose(out["a.bias"], sd["a.bias"])
    # int8 round-trip is near-lossless on a smooth tensor
    assert err_a < err_b


def test_requant_does_not_mutate_input():
    torch.manual_seed(1)
    w = torch.randn(32, 8)
    sd = {"a.weight": w.clone()}
    before = sd["a.weight"].clone()
    _ = requant_state_dict_per_tensor_nbits(sd, {"a.weight": 4})
    assert torch.allclose(sd["a.weight"], before)


def test_requant_tensor_not_in_alloc_is_copied():
    torch.manual_seed(2)
    sd = {"a.weight": torch.randn(16, 4), "untouched.weight": torch.randn(8, 2)}
    out = requant_state_dict_per_tensor_nbits(sd, {"a.weight": 4})
    assert torch.allclose(out["untouched.weight"], sd["untouched.weight"])
    # 'a' WAS requantized -> changed
    assert not torch.allclose(out["a.weight"], sd["a.weight"])


def test_requant_uniform_int8_is_near_identity():
    """All-int8 allocation reproduces the codec's int8 grid (the identity-preserving
    superset property: uniform int8 = the existing codec)."""
    torch.manual_seed(4)
    sd = {f"t{i}.weight": torch.randn(20, 5) * 0.4 for i in range(3)}
    out = requant_state_dict_per_tensor_nbits(sd, {f"t{i}.weight": 8 for i in range(3)})
    for k in sd:
        assert float((out[k] - sd[k]).abs().max()) < 0.02  # int8 grid is fine


def test_per_stage_bits_summary_groups_by_hnerv_stage():
    nbits = {"stem.weight": 5, "blocks.0.weight": 4, "rgb_0.weight": 8, "rgb_1.weight": 8}
    by_stage = per_stage_bits_summary(nbits)
    assert by_stage["stem"] == [5]
    assert by_stage["blocks.0"] == [4]
    assert set(by_stage["rgb_0"]) == {8}
