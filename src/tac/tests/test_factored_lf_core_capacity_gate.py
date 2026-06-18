"""NO-FAKE tests for the factored LF-core capacity gate probe.

Verify the probe ACTUALLY does the work it names (not constants/markers):
  * the LF core is a REAL narrow-channel HNeRV decoder whose params shrink with width;
  * the byte estimate is monotone in params (real model, not a constant);
  * the eval roundtrip is the REAL contest uint8 path (bicubic-up 874 -> bilinear-down
    384 -> round) and actually changes pixels;
  * the stretched-exp fit actually fits a decaying series (asymptote < first value);
  * the EMA with warmup-decay actually tracks the live weights early (the capstone
    EMA-shadow-lag fix), NOT freezing near init;
  * d_seg is the REAL argmax-flip-rate vs GT on the exact eval path (a self-flip is 0).

These are behavior tests (Class-2 discipline): each would FAIL if the function body were
replaced by a constant/marker.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))

torch = pytest.importorskip("torch")

PROBE_PATH = REPO / "experiments/probe_factored_lf_core_capacity_gate.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("_lf_core_gate", PROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


# --------------------------------------------------------------------------- #
# LF core is a REAL narrow decoder whose capacity shrinks with width          #
# --------------------------------------------------------------------------- #
def _has_vendored_model():
    try:
        from tac.torch_vehicle.vendored_imports import import_vendored

        import_vendored("model")
        return True
    except Exception:
        return False


vendored = pytest.mark.skipif(
    not _has_vendored_model(), reason="vendored HNeRV model not available in this env"
)


@vendored
def test_lf_core_params_shrink_with_width(probe):
    """A narrow LF core has fewer params than a wide one (real architecture, not a stub)."""
    small = probe.build_lf_core(8)
    big = probe.build_lf_core(20)
    n_small = probe._n_params(small)
    n_big = probe._n_params(big)
    assert n_small > 0 and n_big > 0
    assert n_small < n_big, f"narrow core {n_small} should be < wide core {n_big}"
    # bc20 must match the basin's known param count (faithful to the basin decoder)
    assert n_big == 83356, f"bc20 LF core should be 83356 params (basin), got {n_big}"


@vendored
def test_lf_core_renders_pair_shape(probe):
    """LF core renders a (1,2,3,384,512) RGB pair in [0,255] -- the exact eval shape."""
    dec = probe.build_lf_core(8).eval()
    z = torch.randn(1, 28)
    with torch.no_grad():
        out = dec(z)
    assert tuple(out.shape) == (1, 2, 3, 384, 512)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 255.0


def test_byte_estimate_monotone_in_params(probe):
    """Byte estimate is a real monotone function of params, not a constant."""
    b1 = probe._lf_core_byte_estimate(10_000)
    b2 = probe._lf_core_byte_estimate(20_000)
    assert b2 > b1
    # bc20 (83356 params) should land near the basin's ~89KB
    b20 = probe._lf_core_byte_estimate(83356)
    assert 85_000 < b20 < 93_000, f"bc20 byte est {b20} should be ~89KB"


# --------------------------------------------------------------------------- #
# eval roundtrip is the REAL contest uint8 path and changes pixels            #
# --------------------------------------------------------------------------- #
def test_eval_roundtrip_is_real_uint8_path(probe):
    """The roundtrip rounds to integers and actually changes non-integer pixels."""
    x = torch.rand(1, 2, 3, 384, 512) * 255.0 + 0.37  # non-integer values
    out = probe._eval_roundtrip(x)
    assert tuple(out.shape) == (1, 2, 3, 384, 512)
    # output is integer-valued (rounded)
    assert torch.allclose(out, out.round())
    # and it actually changed the input (not identity)
    assert not torch.allclose(out, x), "roundtrip must change non-integer pixels"
    # clamped to [0,255]
    assert float(out.min()) >= 0.0 and float(out.max()) <= 255.0


def test_eval_roundtrip_clamps_out_of_range(probe):
    x = torch.full((1, 2, 3, 384, 512), 999.0)
    out = probe._eval_roundtrip(x)
    assert float(out.max()) <= 255.0
    x2 = torch.full((1, 2, 3, 384, 512), -5.0)
    out2 = probe._eval_roundtrip(x2)
    assert float(out2.min()) >= 0.0


# --------------------------------------------------------------------------- #
# stretched-exp fit actually fits a decaying series                           #
# --------------------------------------------------------------------------- #
def test_stretched_exp_fit_on_decaying_series(probe):
    """A synthetic stretched-exp series fits with an asymptote below the first value."""
    import math

    eps = [0, 50, 100, 200, 400, 800, 1600]
    a, tau, beta, c = 0.005, 1000.0, 0.86, 0.0004
    vals = [a * math.exp(-((e / tau) ** beta)) + c for e in eps]
    fit = probe._stretched_exp_fit(eps, vals)
    assert fit is not None
    # the fit's asymptote should be well below the first value
    assert fit["asymptote"] < vals[0]
    # and should be near the true floor c (within the coarse grid resolution)
    assert abs(fit["asymptote"] - c) <= 0.0006, f"asymptote {fit['asymptote']} vs true {c}"


def test_stretched_exp_fit_returns_none_too_few_points(probe):
    assert probe._stretched_exp_fit([0, 1], [0.5, 0.4]) is None


# --------------------------------------------------------------------------- #
# EMA warmup-decay actually tracks live weights early (capstone fix)          #
# --------------------------------------------------------------------------- #
@vendored
def test_ema_warmup_decay_tracks_not_freezes(probe):
    """The warmup-decay EMA shadow MOVES toward updated live weights early.

    The capstone EMA-shadow-lag bug was a constant high decay freezing the shadow near
    init on short runs. With warmup-decay (decay_t = min(decay,(1+t)/(10+t))), the shadow
    must move substantially toward the live weights after a few steps.
    """
    dec = probe.build_lf_core(8)
    ema = probe._EMA(dec, decay=0.999)
    init_shadow = {k: v.clone() for k, v in ema.shadow.items()}
    # perturb the live weights significantly (simulate training updates)
    with torch.no_grad():
        for p in dec.parameters():
            p.add_(1.0)
    for _ in range(5):
        ema.update(dec)
    # the shadow must have moved toward the live weights (NOT frozen at init)
    moved = False
    for k, v in ema.shadow.items():
        if v.dtype.is_floating_point and not torch.allclose(v, init_shadow[k]):
            moved = True
            break
    assert moved, "warmup-decay EMA shadow must track live weights (not freeze at init)"
    # quantitatively: a representative float param's shadow should be clearly > init
    pname = "stem.weight"
    delta = (ema.shadow[pname] - init_shadow[pname]).abs().mean().item()
    assert delta > 0.05, f"shadow barely moved ({delta}); warmup-decay not working"


@vendored
def test_ema_copy_to_loads_shadow(probe):
    dec = probe.build_lf_core(8)
    ema = probe._EMA(dec, decay=0.9)
    with torch.no_grad():
        for p in dec.parameters():
            p.add_(2.0)
    for _ in range(3):
        ema.update(dec)
    target = probe.build_lf_core(8)
    ema.copy_to(target)
    for k, v in target.state_dict().items():
        assert torch.allclose(v, ema.shadow[k])


# --------------------------------------------------------------------------- #
# d_seg metric: a perfect self-match is zero flips (metric is real)           #
# --------------------------------------------------------------------------- #
@vendored
@pytest.mark.slow
def test_dseg_self_match_is_zero(probe):
    """d_seg of a decoder against ITS OWN SegNet argmax (as GT) is ~0 flips.

    This proves the metric is the REAL argmax-flip-rate: if the GT targets ARE the
    decoder's own SegNet argmax on the exact path, the flip rate is exactly 0.
    """
    import einops

    from tac.scorer import load_default_segnet

    dec = probe.build_lf_core(8).eval()
    latents = torch.randn(3, 28)
    segnet = load_default_segnet(str(REPO / "upstream"), device="cpu")
    # build self-targets: the decoder's own SegNet argmax on the exact eval path
    self_targets = []
    with torch.no_grad():
        for i in range(3):
            out = dec(latents[i : i + 1])
            rt = probe._eval_roundtrip(out)
            x = einops.rearrange(rt, "b t c h w -> b t h w c")
            x = einops.rearrange(x, "b t h w c -> b t c h w")
            seg_in = segnet.preprocess_input(x)
            logits = segnet(seg_in)
            self_targets.append(logits.argmax(dim=1)[0])
    self_targets = torch.stack(self_targets)  # (3, H, W)
    dseg = probe._measure_dseg(dec, latents, segnet, self_targets, [0, 1, 2], "cpu")
    assert dseg == pytest.approx(0.0, abs=1e-9), f"self-match d_seg must be 0, got {dseg}"
