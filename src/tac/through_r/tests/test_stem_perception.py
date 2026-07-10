# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.through_r.stem_perception` — read the frozen SegNet texture
perceiver + synthesize minimal per-class texture through R.

Two tiers, mirroring :mod:`test_palette_realization`:

* **$0 pure / stub** — the Phase-1 characterization math (SVD colour direction, opponency,
  DC-fraction low-pass-vs-edge, Nyquist), the Phase-2 texture synth (flat/stripe/checker/gabor),
  the description-length bit accounting, and the measurement plumbing driven by a deterministic
  ``StubSegNet`` (argmax = nearest reference colour). Verifies BEHAVIOUR, not constants.
* **real-SegNet integration** (skipped if the frozen checkpoint is absent) — proves the extraction
  binds the ACTUAL EfficientNet-B2 ``conv_stem`` (32×3×3×3, stride-2, SiLU): the NO-FAKE guard that
  the stub-only suite is not verifying a hollow shell.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.through_r.resolution_chain import SEG_H, SEG_W
from tac.through_r.stem_perception import (
    STEM_STRIDE,
    FilterCharacterization,
    StemFilterBank,
    StemPerceptionError,
    TextureSpec,
    _opponency_label,
    build_default_specs,
    characterize_filters,
    extract_stem_filters,
    make_checker_tile,
    make_flat_tile,
    make_gabor_tile,
    make_stripe_tile,
    measure_tile_responses,
    per_class_price_list,
    stem_nyquist,
    summarize_filter_bank,
    synth_tile,
    texture_dl_bits,
)

# 5 well-separated reference colours (levels 0/255 hit corners exactly).
REF = np.array(
    [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]], dtype=np.float32
)


class StubSegNet:
    """Deterministic SegNet stub: per-pixel argmax = nearest REF colour (logit = -dist²)."""

    def __init__(self, ref: np.ndarray = REF):
        import torch

        self.ref = torch.tensor(np.asarray(ref, dtype=np.float32))  # (5,3)

    def preprocess_input(self, x):  # (b,1,3,H,W) -> (b,3,SEG_H,SEG_W)
        import torch

        x = x[:, -1]
        return torch.nn.functional.interpolate(
            x, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False
        )

    def __call__(self, x):  # (b,3,H,W) -> logits (b,5,H,W)
        xr = x.permute(0, 2, 3, 1)
        d = ((xr[..., None, :] - self.ref[None, None, None]) ** 2).sum(-1)
        return (-d).permute(0, 3, 1, 2)


@pytest.fixture(scope="module")
def stub():
    return StubSegNet()


def _synthetic_bank(weight: np.ndarray) -> StemFilterBank:
    """Wrap a (O,3,3,3) weight array in a StemFilterBank with identity BN (for characterize tests)."""
    o = weight.shape[0]
    return StemFilterBank(
        weight=weight.astype(np.float64),
        bn_weight=np.ones(o),
        bn_bias=np.zeros(o),
        bn_mean=np.zeros(o),
        bn_var=np.ones(o),
        bn_eps=1e-5,
        act="SiLU",
        stride=STEM_STRIDE,
        out_ch=o,
        in_ch=3,
        kh=3,
        kw=3,
    )


# --------------------------------------------------------------------------- #
# PHASE 1 — perceiver characterization (pure numpy).                          #
# --------------------------------------------------------------------------- #
def test_characterize_luminance_lowpass_kernel():
    """A positive all-channel box kernel -> achromatic, high DC-fraction, low-pass kind."""
    box = np.ones((1, 3, 3, 3))  # every channel a constant positive box (pure DC, luminance)
    chars = characterize_filters(_synthetic_bank(box))
    assert len(chars) == 1
    c = chars[0]
    assert c.dc_fraction > 0.9  # pure DC
    assert c.kind == "low-pass (colour/blur)"
    assert "achromatic" in c.opponency
    # colour direction is the (normalized) all-ones luminance axis.
    assert np.allclose(np.abs(c.color_dir), 1 / np.sqrt(3), atol=1e-6)


def test_characterize_zero_mean_edge_kernel_is_high_pass():
    """A zero-sum spatial kernel -> DC-fraction ~0 -> NOT low-pass (edge/high-pass)."""
    edge = np.zeros((1, 3, 3, 3))
    # horizontal derivative on all channels: rows [-1,0,+1] -> zero spatial mean.
    edge[0, :, 0, :] = -1.0
    edge[0, :, 2, :] = 1.0
    chars = characterize_filters(_synthetic_bank(edge))
    c = chars[0]
    assert c.dc_fraction < 0.1
    assert c.kind != "low-pass (colour/blur)"


def test_opponency_label_variants():
    achro = _opponency_label(np.array([1.0, 1.0, 1.0]) / np.sqrt(3))
    assert "achromatic" in achro
    opp = _opponency_label(np.array([0.7, -0.7, 0.0]))
    assert "opponent" in opp and "+R" in opp and "-G" in opp


def test_summarize_filter_bank_aggregates_and_empty_raises():
    w = np.concatenate([np.ones((2, 3, 3, 3)), np.random.default_rng(0).standard_normal((2, 3, 3, 3))])
    summ = summarize_filter_bank(characterize_filters(_synthetic_bank(w)))
    assert summ["n_filters"] == 4
    assert 0.0 <= summ["low_pass_share"] <= 1.0
    assert abs(summ["low_pass_share"] + summ["edge_share"] - 1.0) < 1e-9
    with pytest.raises(StemPerceptionError):
        summarize_filter_bank([])


def test_stem_nyquist_period4_and_stride2():
    ny = stem_nyquist()
    assert ny["stem_stride"] == float(STEM_STRIDE)
    assert ny["finest_period_seg_input_px"] == 4.0
    # camera period is 4 * (cam/seg scale) ~ 9.1 px; the alias wall.
    assert 8.5 < ny["finest_period_camera_px"] < 9.7
    assert ny["nyquist_cycles_per_seg_input_px"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# PHASE 2 — texture synthesis + description length.                           #
# --------------------------------------------------------------------------- #
def test_texture_dl_bits_ordering_and_unknown_raises():
    flat = texture_dl_bits(TextureSpec(family="flat", c_a=(1, 2, 3)))
    checker = texture_dl_bits(TextureSpec(family="checker", c_a=(1, 2, 3), c_b=(4, 5, 6)))
    stripe = texture_dl_bits(TextureSpec(family="stripe", c_a=(1, 2, 3), c_b=(4, 5, 6)))
    # flat (1 colour) < checker (2 colours + period) < stripe (2 colours + period + orient + duty)
    assert flat < checker < stripe
    with pytest.raises(StemPerceptionError):
        texture_dl_bits(TextureSpec(family="nope", c_a=(0, 0, 0)))


def test_synth_flat_is_constant():
    t = make_flat_tile((10.0, 20.0, 30.0))
    assert t.shape == (SEG_H, SEG_W, 3)
    assert np.allclose(t, np.array([10.0, 20.0, 30.0]))


def test_synth_stripe_two_colours_and_period():
    spec = TextureSpec(family="stripe", c_a=(0, 0, 0), c_b=(255, 255, 255), period=8, orientation=0.0)
    t = make_stripe_tile(spec)
    uniq = np.unique(t.reshape(-1, 3), axis=0)
    assert uniq.shape[0] == 2  # exactly two colours
    # along a row the pattern is period-8 (duty 0.5 -> first 4 = c_a, next 4 = c_b).
    row = t[0, :16, 0]
    assert np.array_equal((row > 127).astype(int), [0, 0, 0, 0, 1, 1, 1, 1] * 2)
    with pytest.raises(StemPerceptionError):
        make_stripe_tile(TextureSpec(family="stripe", c_a=(0, 0, 0), period=0))


def test_synth_checker_parity_and_gabor_band_limited():
    ch = make_checker_tile(TextureSpec(family="checker", c_a=(0, 0, 0), c_b=(255, 255, 255), period=1))
    # period-1 checker: adjacent pixels differ (a true checkerboard).
    assert not np.array_equal(ch[0, 0], ch[0, 1])
    gb = make_gabor_tile(TextureSpec(family="gabor", c_a=(0, 0, 0), c_b=(200, 200, 200), period=8))
    # gabor values stay within [c_a, c_b] (band-limited smooth grating, not a hard edge).
    assert gb.min() >= -1e-6 and gb.max() <= 200.0 + 1e-6
    assert 1.0 < gb.std() < 200.0  # genuinely modulated, not flat


def test_build_default_specs_deterministic_and_bounded():
    a = build_default_specs(max_color_pairs=4, seed=7)
    b = build_default_specs(max_color_pairs=4, seed=7)
    assert [s.family for s in a] == [s.family for s in b]  # deterministic
    families = {s.family for s in a}
    assert "flat" in families and "stripe" in families
    # flats cover every colour; the 64-colour default grid => >= 64 flats.
    assert sum(s.family == "flat" for s in a) >= 64


# --------------------------------------------------------------------------- #
# PHASE 2 — measurement plumbing (StubSegNet).                                #
# --------------------------------------------------------------------------- #
def test_measure_tile_responses_stub_nearest_ref(stub):
    """A flat-white tile -> stub argmax picks REF[4]=white (class 4) with positive margin."""
    specs = [TextureSpec(family="flat", c_a=(255, 255, 255)), TextureSpec(family="flat", c_a=(255, 0, 0))]
    resps = measure_tile_responses(stub, specs, through_R=True)
    assert resps[0].modal_class == 4  # white
    assert resps[1].modal_class == 1  # red
    for r in resps:
        assert abs(r.win_fraction.sum() - 1.0) < 1e-9
        assert r.signed_margin[r.modal_class] > 0.0  # winner beats runner-up


def test_per_class_price_list_stub_records_winners_and_flat_floor(stub):
    """The price list finds a cheapest winning spec per reachable class + records the flat floor."""
    pl = per_class_price_list(stub, specs=build_default_specs(max_color_pairs=4, seed=1), through_R=True)
    # white/red/green/blue/black are all reachable flats -> every class has a flat floor recorded.
    assert set(pl.flat_best_margin.keys()) == {"Road", "Lane", "Undrivable", "Movable", "MyCar"}
    # at least one class is won by SOME spec (the stub is a pure colour classifier).
    assert any(v is not None for v in pl.per_class_cheapest.values())
    for _name, pp in pl.per_class_cheapest.items():
        if pp is not None:
            assert pp.margin > 0.0
            assert pp.bits == texture_dl_bits(pp.spec)
    assert "NON-PROMOTABLE" in pl.label


def test_extract_stem_filters_rejects_bad_shape():
    """A fake encoder whose conv_stem is not (O,3,·,·) is refused (shape guard, NO-FAKE)."""
    import torch

    class _FakeConv:
        weight = torch.zeros(8, 1, 3, 3)  # in_ch != 3
        stride = (2, 2)

    class _FakeBN:
        weight = torch.ones(8)
        bias = torch.zeros(8)
        running_mean = torch.zeros(8)
        running_var = torch.ones(8)
        eps = 1e-5
        act = torch.nn.SiLU()

    class _FakeModel:
        conv_stem = _FakeConv()
        bn1 = _FakeBN()

    class _FakeEnc:
        model = _FakeModel()

    class _FakeSeg:
        encoder = _FakeEnc()

    with pytest.raises(StemPerceptionError):
        extract_stem_filters(_FakeSeg())


def test_synth_tile_dispatch_matches_makers():
    spec = TextureSpec(family="stripe", c_a=(0, 0, 0), c_b=(255, 255, 255), period=4)
    assert np.array_equal(synth_tile(spec), make_stripe_tile(spec))
    with pytest.raises(StemPerceptionError):
        synth_tile(TextureSpec(family="bogus", c_a=(0, 0, 0)))


# --------------------------------------------------------------------------- #
# real-SegNet integration (NO-FAKE binding proof; skipped if checkpoint absent).
# --------------------------------------------------------------------------- #
def _real_segnet_or_skip():
    try:
        from tac.boundary_math.seg_core import load_real_segnet

        return load_real_segnet("cpu")
    except (FileNotFoundError, ImportError) as e:  # pragma: no cover - env dependent
        pytest.skip(f"frozen SegNet unavailable: {e}")


def test_real_stem_is_efficientnet_b2_32x3x3x3_stride2():
    seg = _real_segnet_or_skip()
    bank = extract_stem_filters(seg)
    assert (bank.out_ch, bank.in_ch, bank.kh, bank.kw) == (32, 3, 3, 3)
    assert bank.stride == 2
    assert bank.act == "SiLU"
    chars = characterize_filters(bank)
    assert len(chars) == 32
    assert all(isinstance(c, FilterCharacterization) for c in chars)
    summ = summarize_filter_bank(chars)
    # the frozen B2 stem is measured colour/low-pass dominated -> a nonzero low-pass share.
    assert summ["low_pass_share"] > 0.0
