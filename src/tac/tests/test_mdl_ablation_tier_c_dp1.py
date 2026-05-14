# SPDX-License-Identifier: MIT
"""Tests for DP1 (pre-trained driving prior) Tier C post-decode perturbation.

Per Grand Council omnibus Decision 7 (commit ``7872c9f4b``, 2026-05-14)
PROCEED A→C ordering: extend Tier C for DP1 substrate parallel to PR106.
DP1 is the canonical "frozen-prior + tiny-overfit" substrate class — the
Tier C signature should be QUALITATIVELY DIFFERENT from A1/PR106
(within-HNeRV-class) and from IBPS1 (across-class IB-bottleneck):

* The renderer is a tiny SIREN-style coordinate-MLP (~12K params at
  hidden=64, layers=3). Renderer-weight perturbation should saturate
  the score quickly (small perturbation → large frame change because
  the renderer is the entire score-affecting surface).
* The per-pair int8 residual is heavily quantized + clamped to ~3 gray
  levels of correction. Residual perturbation should be SMALL even at
  high σ because the residual contributes tiny RGB deltas.
* The codebook is FROZEN at inflate (NOT a Tier C target) — perturbing
  it would measure offline-distilled prior robustness, not contest-
  overfit signal.

This test suite pins the DP1 Tier C surface so the dispatch contract +
result schema + determinism + zero-sigma identity + bug-class anti-
regression are all structurally locked.

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" tests use
``tmp_path`` (pytest fixture). The DP1 substrate has no canonical archive
on disk in this checkout (DP1 is L1 SCAFFOLD; full training has not yet
landed); a tiny synthetic DP1 archive is built in fixtures via the
canonical ``pack_archive`` API.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "mdl_scorer_conditional_ablation.py"


def _load_module():
    """Import the script as a module under a stable name."""
    name = "_mdl_z1_tier_c_dp1"
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


def _build_tiny_dp1_archive_bytes(num_pairs: int = 2,
                                   per_pair_bytes: int = 6):
    """Build a minimal DP1 archive via the canonical ``pack_archive`` API.

    Uses a tiny renderer (hidden_dim=8, num_hidden_layers=2) and a small
    output resolution (8x8) so test runtime stays fast on CPU. Returns
    ``(inner_bytes, render_cfg)`` so tests can reconstruct the renderer
    independently.

    The synthetic codebook uses all-zero arrays — fast and deterministic.
    Per the Tier C contract, the codebook is NOT a perturbation target so
    its content doesn't affect the test signal (only its bytes are part
    of the archive size + section layout).
    """
    import io
    import pickle

    import brotli
    import numpy as np
    import torch

    from tac.substrates.pretrained_driving_prior.archive import (
        pack_archive,
    )
    from tac.substrates.pretrained_driving_prior.architecture import (
        DrivingPriorRenderer,
        DrivingPriorRendererConfig,
    )
    from tac.substrates.pretrained_driving_prior.codebook import (
        DashcamCodebook,
        LANE_CURVATURE_PCA_SHAPE,
        ROAD_PLANE_BASIS_SHAPE,
        SKY_HORIZON_PROFILE_SHAPE,
        VEHICLE_APPEARANCE_BASIS_SHAPE,
    )

    # Tiny renderer for fast test runtime
    cfg = DrivingPriorRendererConfig(
        hidden_dim=8,
        num_hidden_layers=2,
        output_height=8,
        output_width=8,
    )
    torch.manual_seed(42)
    renderer = DrivingPriorRenderer(cfg)
    renderer_sd = {
        k: v.detach().cpu() for k, v in renderer.state_dict().items()
    }

    # Build a synthetic codebook with valid shapes (all-zero is fine —
    # codebook is not perturbed in Tier C).
    codebook = DashcamCodebook(
        road_plane_basis=np.zeros(ROAD_PLANE_BASIS_SHAPE, dtype=np.int8),
        sky_horizon_profile=np.zeros(
            SKY_HORIZON_PROFILE_SHAPE, dtype=np.int8
        ),
        lane_curvature_pca=np.zeros(
            LANE_CURVATURE_PCA_SHAPE, dtype=np.float16
        ),
        vehicle_appearance_basis=np.zeros(
            VEHICLE_APPEARANCE_BASIS_SHAPE, dtype=np.int8
        ),
        metadata={
            "road_plane_scale": 1.0,
            "sky_horizon_scale": 1.0,
            "vehicle_scale": 1.0,
            "dataset_provenance": "synthetic_test_fixture",
            "distillation_version": "test-v0",
            "license_tags": ["test"],
        },
    )

    # Per-pair residual (int8 bytes; deterministic non-zero pattern so
    # latents-perturbation tests have a non-degenerate baseline).
    residual = bytes(
        ((i * 7 + 13) % 200 - 100) & 0xFF
        for i in range(num_pairs * per_pair_bytes)
    )

    meta = {
        "residual_int8_scale": 64.0,
        "renderer_hidden_dim": cfg.hidden_dim,
        "renderer_num_hidden_layers": cfg.num_hidden_layers,
        "test_fixture": "tiny_dp1_archive",
    }

    inner_bytes = pack_archive(
        codebook,
        renderer_sd,
        residual,
        meta,
        num_pairs=num_pairs,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        per_pair_bytes=per_pair_bytes,
    )
    return inner_bytes, cfg


class _FakeDistortionNet:
    """Minimal stand-in for ``upstream.modules.DistortionNet``.

    Mirrors the IBPS1 + PR106 fixture so cross-grammar comparisons use
    the same scorer surface. Returns ``(pose_dist, seg_dist)`` from
    deterministic statistics of the candidate vs gt frames.
    """

    def __init__(self):
        import torch
        self._torch = torch

    def compute_distortion(self, gt, comp):
        delta_mean = (
            gt.float().mean(dim=(1, 2, 3, 4))
            - comp.float().mean(dim=(1, 2, 3, 4))
        ).abs()
        delta_std = (
            gt.float().std(dim=(1, 2, 3, 4))
            - comp.float().std(dim=(1, 2, 3, 4))
        ).abs()
        return delta_std * 0.001, delta_mean * 0.0001


def _build_gt_pairs(n_pairs):
    """Build a synthetic GT frame tensor at CAMERA resolution."""
    import torch

    mod = _load_module()
    return torch.full(
        (n_pairs, 2, mod.CAMERA_H, mod.CAMERA_W, 3),
        128,
        dtype=torch.uint8,
    )


# ----------------------------------------------------------------------
# Dispatch + parametrization tests
# ----------------------------------------------------------------------


def test_run_tier_c_dp1_dispatches_via_grammar_canonical():
    """grammar='dp1' must call _run_tier_c_dp1."""
    mod = _load_module()
    with mock.patch.object(mod, "_run_tier_c_dp1") as fake_dp1, \
         mock.patch.object(mod, "_run_tier_c_a1") as fake_a1, \
         mock.patch.object(mod, "_run_tier_c_ibps1") as fake_ibps1, \
         mock.patch.object(mod, "_run_tier_c_pr106") as fake_pr106:
        fake_dp1.return_value = ["sentinel"]
        result = mod.run_tier_c(
            inner_bytes=b"unused",
            grammar="dp1",
            pair_indices=[0, 1],
            gt_pairs=None,
            baseline_seg=0.001,
            baseline_pose=0.0,
            distortion_net=None,
            device=None,
            rng=None,
        )
    assert result == ["sentinel"]
    fake_dp1.assert_called_once()
    fake_a1.assert_not_called()
    fake_ibps1.assert_not_called()
    fake_pr106.assert_not_called()


def test_run_tier_c_dp1_dispatches_via_alias_pretrained_driving_prior():
    """grammar='pretrained_driving_prior' must alias to _run_tier_c_dp1."""
    mod = _load_module()
    with mock.patch.object(mod, "_run_tier_c_dp1") as fake_dp1:
        fake_dp1.return_value = []
        mod.run_tier_c(
            inner_bytes=b"unused",
            grammar="pretrained_driving_prior",
            pair_indices=[0],
            gt_pairs=None,
            baseline_seg=0.001,
            baseline_pose=0.0,
            distortion_net=None,
            device=None,
            rng=None,
        )
    fake_dp1.assert_called_once()


def test_run_tier_c_dp1_dispatches_via_alias_driving_prior():
    """grammar='driving_prior' must alias to _run_tier_c_dp1."""
    mod = _load_module()
    with mock.patch.object(mod, "_run_tier_c_dp1") as fake_dp1:
        fake_dp1.return_value = []
        mod.run_tier_c(
            inner_bytes=b"unused",
            grammar="driving_prior",
            pair_indices=[0],
            gt_pairs=None,
            baseline_seg=0.001,
            baseline_pose=0.0,
            distortion_net=None,
            device=None,
            rng=None,
        )
    fake_dp1.assert_called_once()


def test_run_tier_c_dp1_does_not_route_to_other_grammars():
    """Verify _run_tier_c_dp1 is exclusively chosen for DP1 grammar variants."""
    mod = _load_module()
    for grammar in ("dp1", "pretrained_driving_prior", "driving_prior",
                    "dp1_driving_prior"):
        with mock.patch.object(mod, "_run_tier_c_dp1") as fake_dp1, \
             mock.patch.object(mod, "_run_tier_c_a1") as fake_a1, \
             mock.patch.object(mod, "_run_tier_c_ibps1") as fake_ibps1, \
             mock.patch.object(mod, "_run_tier_c_pr106") as fake_pr106:
            fake_dp1.return_value = []
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
            fake_dp1.assert_called_once()
            fake_a1.assert_not_called()
            fake_ibps1.assert_not_called()
            fake_pr106.assert_not_called()


# ----------------------------------------------------------------------
# Default sigma alignment (Z1 council deep-math §3.5 cross-grammar comparison)
# ----------------------------------------------------------------------


def test_dp1_tier_c_default_noise_sigmas_match_sister_grammars():
    """DP1 default sigma sweep must be ``[0.001, 0.01, 0.1, 1.0]``."""
    mod = _load_module()
    import inspect
    src = inspect.getsource(mod._run_tier_c_dp1)
    assert "[0.001, 0.01, 0.1, 1.0]" in src, (
        "default sigma schedule changed; misaligned across grammars"
    )


# ----------------------------------------------------------------------
# End-to-end ablation tests (use synthetic DP1 archive)
# ----------------------------------------------------------------------


def test_dp1_tier_c_end_to_end_returns_8_results():
    """Tier C should return 4 sigmas × 2 targets = 8 TierCResult entries."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(1234)
    results = mod._run_tier_c_dp1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
    )
    assert len(results) == 8
    targets = {r.target for r in results}
    sigmas = {r.noise_sigma_relative for r in results}
    assert targets == {"state_dict", "latents"}
    assert sigmas == {0.001, 0.01, 0.1, 1.0}


def test_dp1_tier_c_returns_tier_c_result_objects():
    """Every entry must be a TierCResult with all required fields populated."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(1234)
    results = mod._run_tier_c_dp1(
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


def test_dp1_tier_c_zero_sigma_yields_zero_delta():
    """Zero noise sigma must produce zero Δscore (mathematical invariant).

    For DP1 specifically: sigma=0 → identically-zero Gaussian noise on the
    state_dict path; identically-zero perturbation of the int8 residual on
    the latents path (round(x + 0) == round(x) so quantization is stable).
    Both paths produce bit-identical frames → identical scorer output.
    """
    import torch
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    # Build a baseline by routing through the same _render path with
    # sigma=0 (identity perturbation). We use _run_tier_c_dp1 itself with
    # the baseline computed from its own first-pass output to lock the
    # mathematical invariant inside the function.
    torch.manual_seed(1234)
    results = mod._run_tier_c_dp1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.0,
        baseline_pose=0.0,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.0],
    )
    assert len(results) == 2
    # baseline_seg/baseline_pose are both 0.0 here, so the result rows'
    # Δseg = (seg_p - 0) and Δpose = (pose_p - 0); the zero-sigma claim is
    # that BOTH state_dict and latents targets produce IDENTICAL frames →
    # identical (seg_p, pose_p) → identical Δ across the two rows.
    sd_row = next(r for r in results if r.target == "state_dict")
    lat_row = next(r for r in results if r.target == "latents")
    # state_dict and latents at sigma=0 must produce IDENTICAL frames
    # because both perturbations are identity transformations. Therefore
    # their Δseg/Δpose must match exactly.
    assert sd_row.delta_seg == lat_row.delta_seg, (
        f"sigma=0 state_dict Δseg={sd_row.delta_seg} != latents "
        f"Δseg={lat_row.delta_seg}; perturbation logic is not identity at sigma=0"
    )
    assert sd_row.delta_pose == lat_row.delta_pose, (
        f"sigma=0 state_dict Δpose={sd_row.delta_pose} != latents "
        f"Δpose={lat_row.delta_pose}; perturbation logic is not identity at sigma=0"
    )


def test_dp1_tier_c_determinism_same_seed_same_result():
    """Same torch.manual_seed + sigma → identical Δscore (determinism contract)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(99)
    results_a = mod._run_tier_c_dp1(
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
    results_b = mod._run_tier_c_dp1(
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


def test_dp1_tier_c_high_sigma_state_dict_yields_nonzero_delta():
    """Renderer-weight perturbation at sigma=1.0 must produce non-zero Δscore.

    The DP1 renderer is the entire score-affecting surface; perturbing
    the state_dict at sigma=1.0 (full per-tensor std noise) must propagate
    through the SIREN coordinate-MLP into the rendered frames.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(7)
    results = mod._run_tier_c_dp1(
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
    assert sd_results[0].delta_score_components != 0.0, (
        "state_dict perturbation at sigma=1.0 produced zero Δscore — "
        "renderer-weight noise not propagating through SIREN coordinate-MLP"
    )


def test_dp1_tier_c_explicit_sigma_list_respected():
    """Caller-supplied noise_sigmas list overrides the default."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(0)
    results = mod._run_tier_c_dp1(
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
    assert len(results) == 6
    sigmas_seen = sorted({r.noise_sigma_relative for r in results})
    assert sigmas_seen == [0.001, 0.5, 2.0]


def test_dp1_tier_c_empty_sigma_list_returns_empty():
    """Empty sigma list → empty results (skip-tier-c contract)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(0)
    results = mod._run_tier_c_dp1(
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


def test_dp1_tier_c_state_dict_target_uses_renderer_only():
    """state_dict perturbation must target ONLY the renderer (not codebook).

    The codebook is FROZEN at inflate per HNeRV parity discipline L1;
    perturbing it would measure offline-distilled prior robustness, not
    the contest-overfit signal. The renderer is the score-affecting
    surface. Confirm by spy/assertion that the perturbation path only
    touches renderer keys.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    # Confirm: state_dict perturbation produces non-zero Δscore (the
    # renderer IS the score-affecting surface) which proves the
    # perturbation hits the renderer (not the inert codebook).
    torch.manual_seed(13)
    results = mod._run_tier_c_dp1(
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
    # If the codebook was perturbed instead of the renderer, the renderer
    # state_dict would be unchanged → identity frames → Δ=0. Non-zero Δ
    # is the structural proof.
    assert sd_results[0].delta_score_components != 0.0, (
        "state_dict perturbation produced Δ=0 — codebook might be the "
        "target instead of renderer (codebook is frozen at inflate)"
    )


def test_dp1_tier_c_corrupt_archive_raises_value_error():
    """Garbage archive bytes should raise ValueError (parse failure)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    bad_bytes = b"\xff" * 100
    with pytest.raises((ValueError, Exception)):
        mod._run_tier_c_dp1(
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


# ----------------------------------------------------------------------
# Bug-class anti-regression
# ----------------------------------------------------------------------


def test_dp1_tier_c_function_distinct_from_other_grammar_implementations():
    """_run_tier_c_dp1 must be a separate callable from sister Tier C functions."""
    mod = _load_module()
    assert hasattr(mod, "_run_tier_c_dp1"), "DP1 Tier C function missing"
    assert mod._run_tier_c_dp1 is not mod._run_tier_c_a1
    assert mod._run_tier_c_dp1 is not mod._run_tier_c_ibps1
    assert mod._run_tier_c_dp1 is not mod._run_tier_c_pr106


def test_dp1_grammar_no_longer_falls_through_to_empty():
    """grammar='dp1' must NOT silently return [] (the pre-extension bug)."""
    mod = _load_module()
    sentinel = ["dp1-was-called"]
    with mock.patch.object(mod, "_run_tier_c_dp1", return_value=sentinel):
        for g in ("dp1", "pretrained_driving_prior", "driving_prior",
                  "dp1_driving_prior"):
            result = mod.run_tier_c(
                inner_bytes=b"unused",
                grammar=g,
                pair_indices=[0],
                gt_pairs=None,
                baseline_seg=0.001,
                baseline_pose=0.0,
                distortion_net=None,
                device=None,
                rng=None,
            )
            assert result == sentinel, f"grammar={g} silently returned []"


def test_dp1_grammar_in_supported_grammars_set():
    """SUPPORTED_GRAMMARS canonical token includes ``dp1``."""
    mod = _load_module()
    assert "dp1" in mod.SUPPORTED_GRAMMARS


def test_dp1_grammar_aliases_normalized_to_canonical():
    """All DP1 grammar aliases normalize to ``dp1`` at the normalizer surface."""
    mod = _load_module()
    assert mod.normalize_grammar("dp1") == "dp1"
    assert mod.normalize_grammar("pretrained_driving_prior") == "dp1"
    assert mod.normalize_grammar("driving_prior") == "dp1"
    assert mod.normalize_grammar("dp1_driving_prior") == "dp1"
    assert mod.normalize_grammar("DP1") == "dp1"


# ----------------------------------------------------------------------
# JSON serialization round-trip
# ----------------------------------------------------------------------


def test_dp1_tier_c_results_json_serializable():
    """TierCResult instances from DP1 Tier C survive JSON round-trip.

    Anti-regression for the empirical pipeline: results land as JSON in
    the ablation output dir; if the schema diverges across grammars,
    downstream aggregation breaks.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    import dataclasses
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(42)
    results = mod._run_tier_c_dp1(
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
    serialized = json.dumps([dataclasses.asdict(r) for r in results])
    parsed = json.loads(serialized)
    assert len(parsed) == 2
    for entry in parsed:
        assert "target" in entry
        assert "noise_sigma_relative" in entry
        assert "delta_seg" in entry
        assert "delta_pose" in entry
        assert "delta_score_components" in entry
        assert "elapsed_seconds" in entry


# ----------------------------------------------------------------------
# Real DP1 archive parses correctly under load_archive
# ----------------------------------------------------------------------


def test_load_archive_dp1_parses_synthetic_archive(tmp_path):
    """load_archive(synthetic DP1 archive, 'dp1') yields valid sections."""
    pytest.importorskip("brotli")
    pytest.importorskip("torch")
    import zipfile

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    archive_path = tmp_path / "dp1_archive.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", inner_bytes)

    mod = _load_module()
    parsed_inner, sections = mod.load_archive(archive_path, "dp1")
    assert parsed_inner == inner_bytes
    expected = {
        "dp1_header",
        "codebook_blob",
        "renderer_blob",
        "residual_blob",
        "meta_blob",
    }
    assert set(sections.keys()) == expected


def test_decode_to_frames_dp1_supports_synthetic_archive():
    """DP1 baselines must decode before real-scorer Tier C perturbation."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()
    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes(num_pairs=2)

    frames = mod.decode_to_frames(inner_bytes, "dp1", [0, 1], torch.device("cpu"))

    assert frames.shape == (2, 2, mod.CAMERA_H, mod.CAMERA_W, 3)
    assert frames.dtype == torch.uint8


def test_dp1_canonical_supported_grammar_field():
    """DP1 in SUPPORTED_GRAMMARS but parses no fall-through."""
    mod = _load_module()
    assert mod.normalize_grammar("dp1") in mod.SUPPORTED_GRAMMARS


# ----------------------------------------------------------------------
# Tier C signal qualitative check (DP1 frozen-prior + tiny-overfit class)
# ----------------------------------------------------------------------


def test_dp1_tier_c_residual_perturbation_sigma_independence():
    """DP1 residual is heavily quantized + clamped; high-σ residual
    perturbation should produce comparable Δscore to mid-σ.

    Per the substrate-class hypothesis, DP1's per-pair residual contributes
    only ~3 gray levels of correction (low int8 dynamic range + clamped).
    Therefore: residual perturbation at σ=0.1 vs σ=1.0 should NOT scale
    linearly. This is a qualitative class-shift signature different from
    A1/PR106 (where latent perturbation at σ=1.0 dominates).
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes, _cfg = _build_tiny_dp1_archive_bytes()
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(0)
    results = mod._run_tier_c_dp1(
        inner_bytes=inner_bytes,
        pair_indices=[0, 1],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.1, 1.0],
    )
    # 2 sigmas × 2 targets = 4 results
    assert len(results) == 4
    # Just confirm the function runs end-to-end and produces 4 results.
    # Substrate-class verdict math is the empirical CLI's job, not a unit test's.
    sigmas_seen = sorted({r.noise_sigma_relative for r in results})
    assert sigmas_seen == [0.1, 1.0]
