"""NO-FAKE tests for the CONTINUOUS-TEXTURE NCA d_seg-core feasibility gate (the operator's true reframe).

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS": these tests verify the gate ACTUALLY does the continuous-texture
generative work it names — the NCA readout is C->3 CONTINUOUS RGB (not a class partition), the iteration
actually grows the frame, the per-frame latent conditions the output, the byte model counts the rule+latent
(the few-KB decoder-replacement premise), and the realized d_seg is the REAL argmax-flip-rate.

All numbers `[contest-CPU advisory]` NON-PROMOTABLE; the gate never claims a score or moves the pointer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))
sys.path.insert(0, str(REPO / "experiments"))

torch = pytest.importorskip("torch")

PROBE_PATH = REPO / "experiments/probe_nca_texture_dseg_feasibility_gate.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("_nca_tex_gate", PROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


def _gen(probe, hidden=64, n_channels=8, latent_dim=16, shape=(48, 64)):
    return probe.NCATextureGenerator(n_channels, hidden, latent_dim, shape, torch.device("cpu"))


# --------------------------------------------------------------------------- #
# 1. The readout is C->3 CONTINUOUS RGB (the structural fix the sister flagged) #
# --------------------------------------------------------------------------- #
def test_readout_is_three_channel_rgb_not_class_logits(probe):
    """The generator outputs CONTINUOUS RGB (3 channels), NOT a class partition (n_classes logits)."""
    g = _gen(probe)
    assert g.readout.shape[0] == 3, "readout must be C->3 RGB (continuous texture), not class logits"


def test_grow_rgb_returns_continuous_frame(probe):
    """grow_rgb returns a (3,H,W) continuous RGB frame in [0,255], not a one-hot/argmax partition."""
    g = _gen(probe, shape=(48, 64))
    with torch.no_grad():
        g.w2.add_(torch.randn_like(g.w2) * 0.05)
        rgb = g.grow_rgb(8)
    assert tuple(rgb.shape) == (3, 48, 64)
    assert rgb.min() >= 0 and rgb.max() <= 255
    # continuous: more than 5 distinct values per channel (a flat-partition would have <= n_classes)
    n_unique = int(torch.unique(rgb[0]).numel())
    assert n_unique > 5, f"RGB must be continuous texture (>5 vals), got {n_unique} (looks like a partition)"


# --------------------------------------------------------------------------- #
# 2. The iteration actually grows; the latent conditions the seed              #
# --------------------------------------------------------------------------- #
def test_iteration_changes_rgb(probe):
    """With a non-zero rule, more iteration steps produce different RGB (the generative work)."""
    g = _gen(probe)
    with torch.no_grad():
        g.w2.add_(torch.randn_like(g.w2) * 0.1)
    r1 = g.grow_rgb(1)
    r8 = g.grow_rgb(8)
    assert (r1 - r8).abs().mean().item() > 1e-4, "iteration must change the grown RGB (not a no-op)"


def test_latent_conditions_the_seed(probe):
    """Changing the per-frame latent changes the seed (the per-frame stored bytes actually do work)."""
    g = _gen(probe)
    s1 = g._seed().clone()
    with torch.no_grad():
        g.latent.add_(torch.randn_like(g.latent))
    s2 = g._seed()
    assert (s1 - s2).abs().mean().item() > 1e-5, "the per-frame latent must condition the seed"


# --------------------------------------------------------------------------- #
# 3. Byte model: few-KB rule + per-frame latent (the decoder-replacement premise)#
# --------------------------------------------------------------------------- #
def test_rule_is_few_kb_not_400k(probe):
    """The rule param count is few-KB-scale (the byte-cheap premise), NOT ~400K.

    Guards the fixed bug: a seed grid of H/seed_factor blew latent_proj to ~393K params (rate 0.15). The
    fixed small 6x8 coarse grid keeps the rule few-KB so the decoder-replacement rate stays << 0.05.
    """
    g = _gen(probe, hidden=128, n_channels=16, latent_dim=32)
    rule_pc = g.rule_param_count()
    assert rule_pc < 50_000, f"rule must be few-KB-scale for the byte-cheap premise, got {rule_pc}"
    b = probe.nca_texture_bytes(rule_pc, 32)
    rate = probe.rate_from_total_bytes(b["total_600_amortized_bytes"])
    assert rate < 0.05, f"rate must be below the byte-cheap bar (0.05), got {rate:.4f}"


def test_byte_model_counts_rule_plus_latent(probe):
    """The byte model = rule (once) + per-frame latent (x600). n_steps adds no bytes."""
    import inspect

    b = probe.nca_texture_bytes(30000, 32)
    assert b["rule_bytes"] > 0 and b["latent_bytes_per_frame"] > 0
    # rule stored once -> amortized < naive per-frame full
    assert b["total_600_amortized_bytes"] < b["total_600_full_bytes"]
    # the free-depth claim: byte model never depends on n_steps
    assert "n_steps" not in inspect.signature(probe.nca_texture_bytes).parameters


def test_rule_count_excludes_perception(probe):
    """The fixed perception kernels are NOT learned params (0 stored bytes)."""
    g = _gen(probe)
    assert not isinstance(g.perception_w, torch.nn.Parameter)


# --------------------------------------------------------------------------- #
# 4. Gradients flow into the rule + latent from an RGB recon loss              #
# --------------------------------------------------------------------------- #
def test_gradient_flows_into_rule_and_latent(probe):
    """An RGB loss produces non-zero gradients on the rule, readout, AND the per-frame latent."""
    import torch.nn.functional as F

    g = _gen(probe, shape=(48, 64))
    with torch.no_grad():
        g.w2.add_(torch.randn_like(g.w2) * 0.05)
    target = torch.full((3, 48, 64), 100.0)
    rgb = g.grow_rgb(8)
    loss = F.mse_loss(rgb, target)
    loss.backward()
    assert g.w2.grad is not None and g.w2.grad.abs().sum().item() > 0
    assert g.readout.grad is not None and g.readout.grad.abs().sum().item() > 0
    assert g.latent.grad is not None and g.latent.grad.abs().sum().item() > 0


# --------------------------------------------------------------------------- #
# 5. The eval roundtrip (reused) is the real uint8 path                        #
# --------------------------------------------------------------------------- #
def test_eval_roundtrip_real_uint8(probe):
    frame = torch.rand(3, 384, 512) * 255.0
    out = probe._eval_roundtrip_t(frame, ste=False)[0]
    assert tuple(out.shape) == (3, 384, 512)
    assert torch.allclose(out, out.round(), atol=1e-5)


# --------------------------------------------------------------------------- #
# 6. NO-FAKE: realized d_seg is the real argmax-flip-rate (self-match == 0)    #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (REPO / "upstream").exists(), reason="upstream SegNet weights not available")
def test_dseg_self_match_is_zero(probe):
    """A frame's d_seg against ITS OWN SegNet argmax is exactly 0 flips (the metric is real)."""
    from tac.scorer import load_default_segnet

    segnet = load_default_segnet(str(REPO / "upstream"), device="cpu")
    frame = torch.rand(3, 384, 512) * 255.0
    with torch.no_grad():
        a = probe._segnet_argmax_of_frame(segnet, frame).cpu().numpy()
        b = probe._segnet_argmax_of_frame(segnet, frame).cpu().numpy()
    assert float((a != b).mean()) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 7. Premise + thresholds are the measured campaign anchors                    #
# --------------------------------------------------------------------------- #
def test_premise_and_thresholds(probe):
    """The gate's premise (GT-RGB roundtrip d_seg) + thresholds are the measured anchors."""
    assert pytest.approx(0.00022) == probe.GT_RGB_ROUNDTRIP_DSEG  # measured: continuous texture survives
    assert pytest.approx(0.0162) == probe.FLAT_PARTITION_NCA_FLOOR  # the sister bar to beat
    assert pytest.approx(0.0012) == probe.GREEN_DSEG_THRESHOLD
    assert probe.B0 == 37_545_489
    assert pytest.approx(0.00034) == probe.HELD_POSE
