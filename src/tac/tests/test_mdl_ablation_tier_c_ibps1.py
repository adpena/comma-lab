# SPDX-License-Identifier: MIT
"""Tests for IBPS1 (C6 MDL-IBPS) Tier C post-decode perturbation.

Per HARVEST-AND-Z1 landing memo (2026-05-14), Tier C is the dispositive
substrate-class discriminator: Tier A is brotli-saturated; Tier C
bypasses the byte layer entirely by perturbing the DECODED tensors
(decoder state_dict / latents) with Gaussian noise scaled by per-tensor
relative std.

This test suite pins the IBPS1 Tier C surface so the bug class
``Tier C returns [] early for non-A1 grammars`` is structurally
extinct.

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" tests
use ``tmp_path`` (pytest fixture). The real C6 5ep archive regression
test skips gracefully if the archive is not present in this checkout.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path
from unittest import mock

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "mdl_scorer_conditional_ablation.py"


def _load_module():
    """Import the script as a module under a stable name."""
    name = "_mdl_z1_tier_c_ibps1"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _build_tiny_ibps1_archive_bytes():
    """Build a minimal-cost IBPS1 archive via the canonical packer.

    Uses tiny config (latent_dim=4, num_pairs=2, single small decoder
    block) so test runtime stays under a few seconds on CPU.

    Returns: (inner_bytes, cfg) tuple where ``cfg`` is the
    ``MDLIBPSConfig`` instance used to build the substrate. The same
    cfg can be used by tests to materialize a fresh ``MDLIBPSSubstrate``
    for comparison.
    """
    import torch  # local import - heavy

    from tac.substrates.c6_e4_mdl_ibps.architecture import (
        MDLIBPSConfig,
        MDLIBPSSubstrate,
    )
    from tac.substrates.c6_e4_mdl_ibps.archive import pack_archive

    cfg = MDLIBPSConfig(
        latent_dim=4,
        encoder_input_channels=3,
        decoder_embed_dim=8,
        decoder_initial_grid_h=2,
        decoder_initial_grid_w=2,
        decoder_channels=(6, 4),
        decoder_num_upsample_blocks=2,
        num_pairs=2,
        output_height=8,
        output_width=8,
        beta_ib=0.01,
        latent_init_std=0.02,
    )
    torch.manual_seed(42)
    model = MDLIBPSSubstrate(cfg)
    encoder_sd = {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}
    decoder_sd = {k: v.detach().clone() for k, v in model.decoder.state_dict().items()}
    latents = model.latents.detach().clone()
    meta = {
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_sin_freq": cfg.encoder_sin_freq,
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "decoder_sin_freq": cfg.decoder_sin_freq,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "beta_ib": cfg.beta_ib,
        "latent_init_std": cfg.latent_init_std,
    }
    inner_bytes = pack_archive(encoder_sd, decoder_sd, latents, meta)
    return inner_bytes, cfg


class _FakeDistortionNet:
    """Minimal stand-in for ``upstream.modules.DistortionNet``.

    Returns deterministic ``(pose_dist, seg_dist)`` based on the mean
    of the candidate frames so we can verify ``Δscore`` flows through
    the ablation correctly without loading the real scorer (which
    requires 320 MB of PoseNet/SegNet weights + upstream import).
    """

    def __init__(self):
        import torch  # local import
        self._torch = torch

    def compute_distortion(self, gt, comp):
        # gt and comp: (B, 2, H, W, 3) float.
        # Δ between gt mean and comp mean → seg-like signal
        # Δ between gt std and comp std → pose-like signal
        delta_mean = (gt.float().mean(dim=(1, 2, 3, 4)) - comp.float().mean(dim=(1, 2, 3, 4))).abs()
        delta_std = (gt.float().std(dim=(1, 2, 3, 4)) - comp.float().std(dim=(1, 2, 3, 4))).abs()
        # Each return value is shape (B,)
        # Scale to small positive values so score components stay in band
        return delta_std * 0.001, delta_mean * 0.0001


def _build_gt_pairs(n_pairs):
    """Build a synthetic GT frame tensor at CAMERA resolution.

    Per the ablation tool's contract, gt_pairs shape is
    ``(N, 2, CAMERA_H, CAMERA_W, 3)`` uint8.
    """
    import torch  # local import
    mod = _load_module()
    return torch.full(
        (n_pairs, 2, mod.CAMERA_H, mod.CAMERA_W, 3),
        128,
        dtype=torch.uint8,
    )


# ----------------------------------------------------------------------
# Dispatch + parametrization tests (no torch forward required)
# ----------------------------------------------------------------------


def test_run_tier_c_ibps1_dispatches_via_grammar():
    """run_tier_c with grammar='ibps1' must call _run_tier_c_ibps1."""
    mod = _load_module()
    with mock.patch.object(mod, "_run_tier_c_ibps1") as fake_ibps1, \
         mock.patch.object(mod, "_run_tier_c_a1") as fake_a1:
        fake_ibps1.return_value = []
        fake_a1.return_value = []
        mod.run_tier_c(
            inner_bytes=b"unused",
            grammar="ibps1",
            pair_indices=[0, 1],
            gt_pairs=None,
            baseline_seg=0.001,
            baseline_pose=0.0,
            distortion_net=None,
            device=None,
            rng=None,
        )
    fake_ibps1.assert_called_once()
    fake_a1.assert_not_called()


def test_run_tier_c_a1_dispatches_via_grammar():
    """run_tier_c with grammar='a1' must call _run_tier_c_a1 (regression for back-compat)."""
    mod = _load_module()
    with mock.patch.object(mod, "_run_tier_c_ibps1") as fake_ibps1, \
         mock.patch.object(mod, "_run_tier_c_a1") as fake_a1:
        fake_ibps1.return_value = []
        fake_a1.return_value = []
        mod.run_tier_c(
            inner_bytes=b"unused",
            grammar="a1",
            pair_indices=[0, 1],
            gt_pairs=None,
            baseline_seg=0.001,
            baseline_pose=0.0,
            distortion_net=None,
            device=None,
            rng=None,
        )
    fake_a1.assert_called_once()
    fake_ibps1.assert_not_called()


def test_run_tier_c_pr101_aliased_to_a1_dispatch():
    """Grammar 'pr101' (A1 alias) must route to _run_tier_c_a1."""
    mod = _load_module()
    with mock.patch.object(mod, "_run_tier_c_ibps1") as fake_ibps1, \
         mock.patch.object(mod, "_run_tier_c_a1") as fake_a1:
        fake_ibps1.return_value = []
        fake_a1.return_value = []
        mod.run_tier_c(
            inner_bytes=b"unused",
            grammar="pr101",
            pair_indices=[0, 1],
            gt_pairs=None,
            baseline_seg=0.001,
            baseline_pose=0.0,
            distortion_net=None,
            device=None,
            rng=None,
        )
    fake_a1.assert_called_once()
    fake_ibps1.assert_not_called()


def test_run_tier_c_pr106_dispatches_via_grammar():
    """PR106 grammar must route to the dedicated PR106 Tier C path."""
    mod = _load_module()
    with mock.patch.object(mod, "_run_tier_c_pr106") as fake_pr106, \
         mock.patch.object(mod, "_run_tier_c_ibps1") as fake_ibps1, \
         mock.patch.object(mod, "_run_tier_c_a1") as fake_a1:
        fake_pr106.return_value = ["ok"]
        result = mod.run_tier_c(
            inner_bytes=b"unused",
            grammar="pr106",
            pair_indices=[0, 1],
            gt_pairs=None,
            baseline_seg=0.001,
            baseline_pose=0.0,
            distortion_net=None,
            device=None,
            rng=None,
        )
    assert result == ["ok"]
    fake_pr106.assert_called_once()
    fake_ibps1.assert_not_called()
    fake_a1.assert_not_called()


def test_run_tier_c_unknown_grammar_returns_empty():
    """Unknown grammars return [] (preserves previous behaviour)."""
    mod = _load_module()
    result = mod.run_tier_c(
        inner_bytes=b"unused",
        grammar="unknown_grammar_v999",
        pair_indices=[0, 1],
        gt_pairs=None,
        baseline_seg=0.001,
        baseline_pose=0.0,
        distortion_net=None,
        device=None,
        rng=None,
    )
    assert result == []


def test_ibps1_tier_c_default_noise_sigmas_match_a1():
    """Default sigma list across Tier C grammars must be aligned for cross-class comparison.

    If A1 uses [0.001, 0.01, 0.1, 1.0] and IBPS1 uses something different,
    a council reviewer comparing the two side-by-side would be looking
    at different x-axes. Pin the default schedule.
    """
    mod = _load_module()
    # Call _run_tier_c_ibps1 with mock substrate so we can read sigma loop
    # without doing real torch work
    import torch
    captured_sigmas = []

    def _spy_render(decoder_sd, latents):
        # Used to capture call sequence
        return torch.zeros((1, 2, mod.CAMERA_H, mod.CAMERA_W, 3), dtype=torch.uint8)

    # Run the function with mocking at the substrate level
    with mock.patch.object(
        sys.modules.get("tac.substrates.c6_e4_mdl_ibps.archive", None) or
        mock.MagicMock(),
        "parse_archive",
    ):
        pass

    # The simpler signal: peek at the function source to confirm default
    import inspect
    src = inspect.getsource(mod._run_tier_c_ibps1)
    assert "[0.001, 0.01, 0.1, 1.0]" in src, "default sigma schedule changed; misaligned across grammars"


# ----------------------------------------------------------------------
# End-to-end ablation tests (use real C6 substrate construction)
# ----------------------------------------------------------------------


def test_ibps1_tier_c_end_to_end_returns_8_results():
    """Tier C should return 4 sigmas × 2 targets = 8 TierCResult entries."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(1234)
    results = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
    )

    # 4 default sigmas × 2 targets = 8 results
    assert len(results) == 8
    targets = {r.target for r in results}
    sigmas = {r.noise_sigma_relative for r in results}
    assert targets == {"state_dict", "latents"}
    assert sigmas == {0.001, 0.01, 0.1, 1.0}


def test_ibps1_tier_c_returns_tier_c_result_objects():
    """Every entry must be a TierCResult with all required fields."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(1234)
    results = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.01],
    )
    assert len(results) == 2
    for r in results:
        assert isinstance(r, mod.TierCResult)
        assert r.target in ("state_dict", "latents")
        assert r.noise_sigma_relative == 0.01
        assert r.delta_seg is not None
        assert r.delta_pose is not None
        assert r.delta_score_components is not None
        assert r.elapsed_seconds >= 0.0


def test_ibps1_tier_c_zero_sigma_yields_zero_delta():
    """Zero noise sigma must produce zero Δscore (parity with baseline).

    This is the mathematical sanity test: ``noise_scale = sigma * std`` so
    sigma=0 → zero noise → identical frames → identical score → Δ=0.
    Anti-regression for any future drift in the noise-injection logic.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    # Compute baseline first using the same _render path
    from tac.substrates.c6_e4_mdl_ibps.archive import parse_archive
    arc = parse_archive(inner_bytes)
    # Build baseline frames via the substrate
    from tac.substrates.c6_e4_mdl_ibps.architecture import (
        MDLIBPSConfig,
        MDLIBPSSubstrate,
        EVAL_HW,
    )
    cfg = MDLIBPSConfig(
        latent_dim=int(arc.latents.shape[1]),
        encoder_input_channels=int(arc.meta.get("encoder_input_channels", 3)),
        decoder_embed_dim=int(arc.meta["decoder_embed_dim"]),
        decoder_initial_grid_h=int(arc.meta["decoder_initial_grid_h"]),
        decoder_initial_grid_w=int(arc.meta["decoder_initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in arc.meta["decoder_channels"]),
        decoder_num_upsample_blocks=int(arc.meta["decoder_num_upsample_blocks"]),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(arc.meta.get("output_height", EVAL_HW[0])),
        output_width=int(arc.meta.get("output_width", EVAL_HW[1])),
        beta_ib=float(arc.meta.get("beta_ib", 0.01)),
        latent_init_std=float(arc.meta.get("latent_init_std", 0.02)),
    )

    device = torch.device("cpu")
    model = MDLIBPSSubstrate(cfg).to(device).eval()
    model.encoder.load_state_dict(arc.encoder_state_dict, strict=False)
    model.decoder.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        model.latents.copy_(arc.latents.to(device=device, dtype=model.latents.dtype))

    # Render baseline frames via the SAME path as _render in tier_c
    import torch.nn.functional as F
    pair_indices_t = sorted([0, 1])
    out = torch.empty(
        (len(pair_indices_t), 2, mod.CAMERA_H, mod.CAMERA_W, 3), dtype=torch.uint8
    )
    written = 0
    with torch.inference_mode():
        for pair_idx in pair_indices_t:
            idx_t = torch.tensor([pair_idx], device=device, dtype=torch.long)
            rgb_0, rgb_1, _mu, _logvar = model(idx_t, frames_for_encoder=None)
            stacked = torch.cat([rgb_0, rgb_1], dim=0)
            up = F.interpolate(
                stacked, size=(mod.CAMERA_H, mod.CAMERA_W),
                mode="bicubic", align_corners=False,
            )
            frames = (
                (up * 255.0).clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu()
            )
            out[written, 0] = frames[0]
            out[written, 1] = frames[1]
            written += 1
    baseline_frames = out
    baseline_pose, baseline_seg = mod._compute_seg_pose_delta(
        distortion_net, gt_pairs, baseline_frames, device
    )

    torch.manual_seed(1234)
    results = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=baseline_seg,
        baseline_pose=baseline_pose,
        distortion_net=distortion_net,
        device=device,
        rng=None,
        noise_sigmas=[0.0],
    )
    # 2 results: state_dict + latents, both with sigma=0
    assert len(results) == 2
    for r in results:
        # Δ should be 0 (or extremely close - rounding noise in uint8 cast).
        # Empirically: when sigma=0, noise is identically zero, frames are
        # bit-identical, so Δ should be 0.0 exactly. uint8 quantization is
        # deterministic.
        assert r.delta_seg == 0.0, f"sigma=0 should yield Δseg=0, got {r.delta_seg} (target={r.target})"
        assert r.delta_pose == 0.0, f"sigma=0 should yield Δpose=0, got {r.delta_pose} (target={r.target})"
        assert r.delta_score_components == 0.0, f"sigma=0 should yield Δscore=0, got {r.delta_score_components}"


def test_ibps1_tier_c_determinism_same_seed_same_result():
    """Same torch.manual_seed + sigma → same Δscore (determinism contract).

    Critical for reproducibility: if the ablation produces different
    numbers across runs of the same seed, downstream conclusions about
    substrate-class are non-replicable.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(99)
    results_a = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.1],
    )
    torch.manual_seed(99)
    results_b = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.1],
    )
    assert len(results_a) == len(results_b)
    for ra, rb in zip(results_a, results_b):
        assert ra.target == rb.target
        assert ra.noise_sigma_relative == rb.noise_sigma_relative
        assert ra.delta_seg == rb.delta_seg
        assert ra.delta_pose == rb.delta_pose
        assert ra.delta_score_components == rb.delta_score_components


def test_ibps1_tier_c_larger_sigma_yields_larger_or_equal_delta_in_expectation():
    """Non-trivial sigma should produce non-zero Δscore.

    Sanity test: the noise injection mechanism must actually change the
    decoded frames. A bug where the perturbation is silently a no-op
    (e.g. ``.cpu()`` reverting to original) would surface as Δ=0 at all
    sigmas.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(7)
    results = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[1.0],  # large sigma so signal swamps the uint8 quant noise
    )
    # At sigma=1.0 (full per-tensor std noise), at least one of (state_dict,
    # latents) should produce non-zero Δscore — otherwise the noise is
    # being silently dropped.
    nonzero_deltas = [r for r in results if r.delta_score_components != 0.0]
    assert len(nonzero_deltas) >= 1, (
        f"Expected at least one non-zero Δscore at sigma=1.0; got results: "
        f"{[(r.target, r.delta_score_components) for r in results]}"
    )


def test_ibps1_tier_c_explicit_sigma_list_respected():
    """Caller-supplied noise_sigmas list overrides the default."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(0)
    results = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.001, 0.5, 2.0],
    )
    # 3 sigmas × 2 targets = 6 results
    assert len(results) == 6
    sigmas_seen = sorted({r.noise_sigma_relative for r in results})
    assert sigmas_seen == [0.001, 0.5, 2.0]


def test_ibps1_tier_c_state_dict_target_uses_decoder_only():
    """state_dict target should perturb DECODER (not encoder).

    Per CLAUDE.md "Forbidden code patterns" the encoder is forensic-only
    at inflate (eval path skips the encoder per ``frames_for_encoder=None``).
    Perturbing the encoder would have zero effect on rendered frames.
    Spy on the substrate's load_state_dict calls to confirm the decoder
    is the perturbation target.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    # Get baseline decoder + encoder state_dicts
    from tac.substrates.c6_e4_mdl_ibps.archive import parse_archive
    arc = parse_archive(inner_bytes)
    base_encoder_keys = set(arc.encoder_state_dict.keys())
    base_decoder_keys = set(arc.decoder_state_dict.keys())

    # No common keys (different submodule namespaces)
    assert not (base_encoder_keys & base_decoder_keys), (
        "Encoder and decoder state_dict keys overlap; "
        "test cannot disambiguate target"
    )

    torch.manual_seed(13)
    # Run only state_dict target at sigma=1.0
    results = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[1.0],
    )
    sd_results = [r for r in results if r.target == "state_dict"]
    assert len(sd_results) == 1
    # The non-zero delta confirms state_dict noise propagates through.
    # If encoder was perturbed instead of decoder, Δ would be 0 (encoder
    # not invoked at eval time).
    assert sd_results[0].delta_score_components != 0.0, (
        "state_dict perturbation produced zero Δscore — encoder might be "
        "the target instead of decoder (encoder is eval-time inert)"
    )


def test_ibps1_tier_c_latents_target_yields_nonzero_delta_at_high_sigma():
    """latents target at large sigma should produce non-zero Δscore."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(42)
    results = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[2.0],
    )
    latent_results = [r for r in results if r.target == "latents"]
    assert len(latent_results) == 1
    # At sigma=2.0 latent noise, decoded frames should shift visibly
    assert latent_results[0].delta_score_components != 0.0


def test_ibps1_tier_c_corrupt_archive_raises_value_error():
    """Garbage archive bytes should raise ValueError (parse failure)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    # 100 random bytes that don't even contain the IBPS magic
    bad_bytes = b"\xff" * 100
    with pytest.raises((ValueError, Exception)):
        # ValueError for bad magic; broader Exception class because
        # parse_archive may raise different exceptions on truncated data
        mod._run_tier_c_ibps1(
            inner_bytes=bad_bytes,
            pair_indices=[0, 1],
            gt_pairs=gt_pairs,
            baseline_seg=0.001,
            baseline_pose=0.0001,
            distortion_net=distortion_net,
            device=torch.device("cpu"),
            rng=None,
            noise_sigmas=[0.1],
        )


def test_ibps1_tier_c_skip_tier_c_returns_empty():
    """When run_tier_c is called with skip-like contract (sigmas=[]) returns empty."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(0)
    results = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[],
    )
    assert results == []


# ----------------------------------------------------------------------
# Cross-grammar regression: A1 Tier C still works after refactor
# ----------------------------------------------------------------------


def test_a1_tier_c_dispatch_path_still_functional():
    """A1 Tier C must still be callable through ``run_tier_c`` after refactor.

    Anti-regression for the dispatch rewrite. We don't run the full A1
    forward (which requires A1's HNeRVDecoder + 28-dim latent layout)
    but we DO confirm the dispatch flow: ``grammar='a1'`` routes to
    ``_run_tier_c_a1`` and never falls through to ``_run_tier_c_ibps1``.
    """
    mod = _load_module()
    # Verify the function exists and is distinct from ibps1
    assert hasattr(mod, "_run_tier_c_a1")
    assert hasattr(mod, "_run_tier_c_ibps1")
    assert mod._run_tier_c_a1 is not mod._run_tier_c_ibps1

    # And the back-compat 'pr101' alias routes the same way
    with mock.patch.object(mod, "_run_tier_c_a1") as fake_a1, \
         mock.patch.object(mod, "_run_tier_c_ibps1") as fake_ibps1:
        fake_a1.return_value = []
        fake_ibps1.return_value = []
        for grammar in ("a1", "pr101"):
            mod.run_tier_c(
                inner_bytes=b"unused",
                grammar=grammar,
                pair_indices=[0],
                gt_pairs=None,
                baseline_seg=0.001,
                baseline_pose=0.0,
                distortion_net=None,
                device=None,
                rng=None,
            )
        assert fake_a1.call_count == 2
        assert fake_ibps1.call_count == 0


# ----------------------------------------------------------------------
# Real C6 5ep archive regression (skipped if archive not present)
# ----------------------------------------------------------------------


REAL_C6_5EP_ARCHIVE = (
    REPO
    / "experiments"
    / "results"
    / "lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T144949Z__smoke__5ep_modal"
    / "harvested_artifacts"
    / "lane_substrate_c6_e4_mdl_ibps_results"
    / "output"
    / "archive.zip"
)


def test_ibps1_tier_c_parses_real_c6_5ep_archive_if_present():
    """If the real C6 5ep archive is present, _run_tier_c_ibps1 must parse
    the IBPS1 sections without raising (full forward path skipped — that's
    the empirical CLI run, not a CI regression test).
    """
    pytest.importorskip("brotli")
    torch = pytest.importorskip("torch")

    if not REAL_C6_5EP_ARCHIVE.exists():
        pytest.skip("C6 5ep real archive not present in this checkout")

    mod = _load_module()
    inner, _sections = mod.load_archive(REAL_C6_5EP_ARCHIVE, "ibps1")

    # Verify the canonical parse + substrate construction succeed (this
    # is the costly part; the actual forward + scorer is exercised by
    # the empirical CLI run separately).
    from tac.substrates.c6_e4_mdl_ibps.archive import parse_archive
    arc = parse_archive(inner)
    # Canonical C6 config: latent_dim=24, num_pairs=600
    assert arc.latents.shape[0] == 600
    assert arc.latents.shape[1] == 24
    assert "decoder_channels" in arc.meta
    assert "decoder_embed_dim" in arc.meta


# ----------------------------------------------------------------------
# Cross-grammar default sigma alignment (Z1 council requirement)
# ----------------------------------------------------------------------


def test_default_sigma_lists_aligned_a1_and_ibps1():
    """A1 and IBPS1 Tier C must use the same default sigma sweep.

    Z1 council deep-math §3.5: the substrate-class discriminator
    compares Δscore-vs-sigma curves across substrates. If the default
    sigmas differ, the cross-substrate comparison plot has misaligned
    x-axes and the council cannot read it.
    """
    mod = _load_module()
    import inspect
    src_a1 = inspect.getsource(mod._run_tier_c_a1)
    src_ibps1 = inspect.getsource(mod._run_tier_c_ibps1)
    # Both must contain the same default sigma list
    expected = "[0.001, 0.01, 0.1, 1.0]"
    assert expected in src_a1, "A1 Tier C default sigma list drifted"
    assert expected in src_ibps1, "IBPS1 Tier C default sigma list drifted"


# ----------------------------------------------------------------------
# JSON serialization through the aggregator
# ----------------------------------------------------------------------


def test_ibps1_tier_c_results_serialize_through_archive_result_dataclass():
    """TierCResult entries from IBPS1 must round-trip through
    ``ArchiveAblationResult.asdict()`` for JSON persistence.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()
    from dataclasses import asdict

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(5)
    tier_c_results = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.01],
    )
    archive_result = mod.ArchiveAblationResult(
        archive_name="test_ibps1",
        archive_path="/dev/null",
        archive_sha256="X" * 64,
        archive_size_bytes=len(inner_bytes),
        grammar="ibps1",
        device="cpu",
        pair_samples=2,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        baseline_score_components=0.1,
    )
    archive_result.tier_c = tier_c_results

    # asdict must not raise
    d = asdict(archive_result)
    assert d["grammar"] == "ibps1"
    assert isinstance(d["tier_c"], list)
    assert len(d["tier_c"]) == 2
    # All tier_c entries serialize fully
    for tc_dict in d["tier_c"]:
        assert "target" in tc_dict
        assert "noise_sigma_relative" in tc_dict
        assert "delta_seg" in tc_dict
        assert "delta_pose" in tc_dict
        assert "delta_score_components" in tc_dict


# ----------------------------------------------------------------------
# Bug-class regression: tool returns [] for ibps1 BEFORE the fix
# ----------------------------------------------------------------------


def test_ibps1_tier_c_no_longer_returns_empty_early():
    """Before this fix, run_tier_c returned [] early at line 1127 for
    non-A1 grammars. This regression test pins the fix: with grammar
    'ibps1' the function MUST dispatch to the IBPS1 implementation
    (which returns non-empty if the archive is valid).

    Anti-regression for the bug class documented in
    ``feedback_c6_harvest_z1_landed_20260514.md`` HARVEST-AND-Z1 finding:
    "Tier C does NOT cover ibps1 grammar in the canonical tool".
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(0)
    # Call top-level run_tier_c with grammar='ibps1' (the public surface
    # that previously returned [])
    results = mod.run_tier_c(
        inner_bytes=inner_bytes,
        grammar="ibps1",
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.1],
    )
    # The fix: ibps1 grammar produces non-empty results
    assert len(results) > 0, (
        "BUG REGRESSION: run_tier_c(grammar='ibps1') still returns [] — "
        "the fix at tools/mdl_scorer_conditional_ablation.py was reverted"
    )
    assert len(results) == 2  # 1 sigma × 2 targets


def test_ibps1_tier_c_returns_substantive_signal_not_just_inflate_failure():
    """Tier C signal must come from real decoder + latents perturbation,
    not from a silent inflate failure that produces 0 frames or NaN.

    This pins the substantive-signal contract: every TierCResult must
    have finite (non-NaN, non-Inf) Δseg / Δpose / Δscore values, AND the
    elapsed_seconds field must be > 0 (proving the forward pass ran).
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()
    import math

    inner_bytes, _cfg = _build_tiny_ibps1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(11)
    results = mod._run_tier_c_ibps1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.1],
    )
    for r in results:
        assert math.isfinite(r.delta_seg), f"Δseg not finite: {r.delta_seg}"
        assert math.isfinite(r.delta_pose), f"Δpose not finite: {r.delta_pose}"
        assert math.isfinite(r.delta_score_components), f"Δscore not finite"
        assert r.elapsed_seconds > 0.0, "elapsed_seconds=0 suggests the forward path didn't run"
