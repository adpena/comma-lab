# SPDX-License-Identifier: MIT
"""Tests for PR106 (latent_sidecar_r2) Tier C post-decode perturbation.

Per Grand Council omnibus Decision 7 (commit ``7872c9f4b``, 2026-05-14)
PROCEED A→C ordering: extend Tier C for PR106_latent_sidecar so the
substrate-class discriminator can compare A1 (within-HNeRV-class) vs PR106
(within-HNeRV-class with latent sidecar) vs IBPS1 (across-class IB
bottleneck) vs DP1 (across-class frozen-prior + tiny-overfit) on a single
plot.

This test suite pins the PR106 Tier C surface so the dispatch contract +
result schema + determinism + zero-sigma identity are all structurally
locked. The PR106 _run_tier_c_pr106 implementation already landed in
commit ``c4938a25f`` (the AUTOPILOT-TIER-C-INTEGRATION subagent landing);
this file fills the missing test surface.

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" tests use
``tmp_path`` (pytest fixture). The real PR106 canonical archive
(``submissions/pr106_latent_sidecar_r2/archive.zip``) is used as the live
integration fixture; tests that depend on it skip gracefully if the
archive is not present.

Tests are split into:

* **Dispatch tests**: pin ``run_tier_c(grammar='pr106*')`` routes to
  ``_run_tier_c_pr106``; pin alias forms (``pr106``,
  ``pr106_latent_sidecar``, ``pr106_latent_sidecar_r2``).
* **Default sigma alignment**: pin the ``[0.001, 0.01, 0.1, 1.0]``
  schedule alignment with A1 / IBPS1 / DP1.
* **End-to-end**: run the full forward path on the canonical PR106 archive
  with a tiny pair_indices subset + a fake distortion net — verify the
  result shape, schema, and substantive Δscore at sigma=1.0.
* **Bug-class anti-regression**: verify ``grammar='pr106'`` no longer
  silently returns [] (the old pre-c4938a25f behavior); verify
  ``_run_tier_c_pr106`` exists and is distinct from sister functions.
* **JSON serialization round-trip**: Tier C results survive
  ``ArchiveAblationResult`` serialization.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "mdl_scorer_conditional_ablation.py"
CANONICAL_PR106_ARCHIVE = (
    REPO / "submissions" / "pr106_latent_sidecar_r2" / "archive.zip"
)


def _load_module():
    """Import the script as a module under a stable name."""
    name = "_mdl_z1_tier_c_pr106"
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


class _FakeDistortionNet:
    """Minimal stand-in for ``upstream.modules.DistortionNet``.

    Mirrors the IBPS1 test fixture so cross-grammar comparisons use the
    same scorer surface during ablation. Returns deterministic
    ``(pose_dist, seg_dist)`` from the candidate vs gt frame statistics.
    """

    def __init__(self):
        import torch  # local import — heavy

        self._torch = torch

    def compute_distortion(self, gt, comp):
        # gt and comp: (B, 2, H, W, 3) float.
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


def _read_pr106_inner_bytes() -> bytes | None:
    """Return inner blob bytes from the canonical PR106 archive (or None if absent)."""
    if not CANONICAL_PR106_ARCHIVE.is_file():
        return None
    mod = _load_module()
    inner_bytes, _sections = mod.load_archive(
        CANONICAL_PR106_ARCHIVE, "pr106_latent_sidecar"
    )
    return inner_bytes


# ----------------------------------------------------------------------
# Dispatch + parametrization tests (no torch forward required)
# ----------------------------------------------------------------------


def test_run_tier_c_pr106_dispatches_via_grammar_canonical():
    """grammar='pr106_latent_sidecar' must call _run_tier_c_pr106."""
    mod = _load_module()
    with mock.patch.object(mod, "_run_tier_c_pr106") as fake_pr106, \
         mock.patch.object(mod, "_run_tier_c_a1") as fake_a1, \
         mock.patch.object(mod, "_run_tier_c_ibps1") as fake_ibps1:
        fake_pr106.return_value = ["sentinel"]
        result = mod.run_tier_c(
            inner_bytes=b"unused",
            grammar="pr106_latent_sidecar",
            pair_indices=[0, 1],
            gt_pairs=None,
            baseline_seg=0.001,
            baseline_pose=0.0,
            distortion_net=None,
            device=None,
            rng=None,
        )
    assert result == ["sentinel"]
    fake_pr106.assert_called_once()
    fake_a1.assert_not_called()
    fake_ibps1.assert_not_called()


def test_run_tier_c_pr106_dispatches_via_grammar_alias_pr106():
    """grammar='pr106' must alias to _run_tier_c_pr106."""
    mod = _load_module()
    with mock.patch.object(mod, "_run_tier_c_pr106") as fake_pr106:
        fake_pr106.return_value = []
        mod.run_tier_c(
            inner_bytes=b"unused",
            grammar="pr106",
            pair_indices=[0],
            gt_pairs=None,
            baseline_seg=0.001,
            baseline_pose=0.0,
            distortion_net=None,
            device=None,
            rng=None,
        )
    fake_pr106.assert_called_once()


def test_run_tier_c_pr106_dispatches_via_grammar_alias_r2():
    """grammar='pr106_latent_sidecar_r2' (the canonical r2 variant alias) routes to _run_tier_c_pr106."""
    mod = _load_module()
    with mock.patch.object(mod, "_run_tier_c_pr106") as fake_pr106:
        fake_pr106.return_value = []
        mod.run_tier_c(
            inner_bytes=b"unused",
            grammar="pr106_latent_sidecar_r2",
            pair_indices=[0],
            gt_pairs=None,
            baseline_seg=0.001,
            baseline_pose=0.0,
            distortion_net=None,
            device=None,
            rng=None,
        )
    fake_pr106.assert_called_once()


def test_run_tier_c_pr106_does_not_route_to_other_grammars():
    """Verify _run_tier_c_pr106 is exclusively chosen for pr106 grammar variants."""
    mod = _load_module()
    for grammar in ("pr106", "pr106_latent_sidecar", "pr106_latent_sidecar_r2"):
        with mock.patch.object(mod, "_run_tier_c_pr106") as fake_pr106, \
             mock.patch.object(mod, "_run_tier_c_a1") as fake_a1, \
             mock.patch.object(mod, "_run_tier_c_ibps1") as fake_ibps1, \
             mock.patch.object(mod, "_run_tier_c_dp1") as fake_dp1:
            fake_pr106.return_value = []
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
            fake_pr106.assert_called_once()
            fake_a1.assert_not_called()
            fake_ibps1.assert_not_called()
            fake_dp1.assert_not_called()


# ----------------------------------------------------------------------
# Default sigma alignment (Z1 council deep-math §3.5 cross-grammar comparison)
# ----------------------------------------------------------------------


def test_pr106_tier_c_default_noise_sigmas_match_a1_and_ibps1():
    """PR106 default sigma sweep must be ``[0.001, 0.01, 0.1, 1.0]``.

    Per Z1 council deep-math §3.5: the substrate-class discriminator
    compares Δscore-vs-sigma curves across substrates. If the default
    sigmas differ across grammars, the cross-substrate comparison plot
    has misaligned x-axes and the council cannot read it.
    """
    mod = _load_module()
    import inspect
    src = inspect.getsource(mod._run_tier_c_pr106)
    assert "[0.001, 0.01, 0.1, 1.0]" in src, (
        "default sigma schedule changed; misaligned across grammars"
    )


def test_default_sigma_lists_aligned_a1_pr106_ibps1_dp1():
    """A1 + PR106 + IBPS1 + DP1 Tier C must use the same default sigma sweep."""
    mod = _load_module()
    import inspect
    src_a1 = inspect.getsource(mod._run_tier_c_a1)
    src_pr106 = inspect.getsource(mod._run_tier_c_pr106)
    src_ibps1 = inspect.getsource(mod._run_tier_c_ibps1)
    src_dp1 = inspect.getsource(mod._run_tier_c_dp1)
    sigma_token = "[0.001, 0.01, 0.1, 1.0]"
    assert sigma_token in src_a1, "A1 default sigmas drifted"
    assert sigma_token in src_pr106, "PR106 default sigmas drifted"
    assert sigma_token in src_ibps1, "IBPS1 default sigmas drifted"
    assert sigma_token in src_dp1, "DP1 default sigmas drifted"


# ----------------------------------------------------------------------
# End-to-end ablation tests (use canonical PR106 archive when present)
# ----------------------------------------------------------------------


def test_pr106_tier_c_end_to_end_returns_8_results():
    """Tier C should return 4 sigmas × 2 targets = 8 TierCResult entries."""
    if not CANONICAL_PR106_ARCHIVE.is_file():
        pytest.skip("canonical PR106 archive not present in this checkout")
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes = _read_pr106_inner_bytes()
    assert inner_bytes is not None
    gt_pairs = _build_gt_pairs(2)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(1234)
    results = mod._run_tier_c_pr106(
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


def test_pr106_tier_c_returns_tier_c_result_objects():
    """Every entry must be a TierCResult with all required fields populated."""
    if not CANONICAL_PR106_ARCHIVE.is_file():
        pytest.skip("canonical PR106 archive not present in this checkout")
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes = _read_pr106_inner_bytes()
    assert inner_bytes is not None
    gt_pairs = _build_gt_pairs(1)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(1234)
    results = mod._run_tier_c_pr106(
        inner_bytes=inner_bytes,
        pair_indices=[0],
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


def test_pr106_tier_c_zero_sigma_yields_zero_delta():
    """Zero noise sigma must produce zero Δscore (mathematical invariant).

    ``noise_scale = sigma * std`` so sigma=0 → identically-zero noise →
    bit-identical frames → identical scorer output → Δ=0. This is the
    canonical sanity test for the noise-injection logic; a regression
    here means the perturbation is leaking some non-zero signal at zero
    sigma (e.g. dtype cast drift, brotli re-encode noise).
    """
    if not CANONICAL_PR106_ARCHIVE.is_file():
        pytest.skip("canonical PR106 archive not present in this checkout")
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes = _read_pr106_inner_bytes()
    assert inner_bytes is not None
    gt_pairs = _build_gt_pairs(1)
    distortion_net = _FakeDistortionNet()

    # Compute baseline frames using the SAME _render_pr106_components path
    # as Tier C — so any drift between baseline rendering and Tier C
    # rendering is captured (the test is the regression for that drift).
    state_dict, latents = mod._decode_pr106_components(inner_bytes)
    _codec_mod, model_mod = mod._load_pr106_runtime_modules()
    decoder = mod._make_pr106_decoder(
        model_mod.HNeRVDecoder, state_dict, latents, torch.device("cpu")
    )
    baseline_frames = mod._render_pr106_components(
        decoder, latents.to(torch.device("cpu")), [0], torch.device("cpu")
    )
    baseline_pose, baseline_seg = mod._compute_seg_pose_delta(
        distortion_net, gt_pairs, baseline_frames, torch.device("cpu")
    )

    torch.manual_seed(1234)
    results = mod._run_tier_c_pr106(
        inner_bytes=inner_bytes,
        pair_indices=[0],
        gt_pairs=gt_pairs,
        baseline_seg=baseline_seg,
        baseline_pose=baseline_pose,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.0],
    )
    assert len(results) == 2
    for r in results:
        assert r.delta_seg == 0.0, (
            f"sigma=0 should yield Δseg=0, got {r.delta_seg} (target={r.target})"
        )
        assert r.delta_pose == 0.0, (
            f"sigma=0 should yield Δpose=0, got {r.delta_pose} (target={r.target})"
        )
        assert r.delta_score_components == 0.0, (
            f"sigma=0 should yield Δscore=0, got {r.delta_score_components}"
        )


def test_pr106_tier_c_determinism_same_seed_same_result():
    """Same torch.manual_seed + sigma → identical Δscore (determinism contract).

    Critical for reproducibility: substrate-class verdicts are derived
    from Tier C curves; non-determinism would make the verdict
    non-replicable.
    """
    if not CANONICAL_PR106_ARCHIVE.is_file():
        pytest.skip("canonical PR106 archive not present in this checkout")
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes = _read_pr106_inner_bytes()
    assert inner_bytes is not None
    gt_pairs = _build_gt_pairs(1)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(99)
    results_a = mod._run_tier_c_pr106(
        inner_bytes=inner_bytes,
        pair_indices=[0],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[0.1],
    )
    torch.manual_seed(99)
    results_b = mod._run_tier_c_pr106(
        inner_bytes=inner_bytes,
        pair_indices=[0],
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


def test_pr106_tier_c_high_sigma_yields_nonzero_delta():
    """Non-trivial sigma should produce non-zero Δscore (perturbation propagates)."""
    if not CANONICAL_PR106_ARCHIVE.is_file():
        pytest.skip("canonical PR106 archive not present in this checkout")
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes = _read_pr106_inner_bytes()
    assert inner_bytes is not None
    gt_pairs = _build_gt_pairs(1)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(7)
    results = mod._run_tier_c_pr106(
        inner_bytes=inner_bytes,
        pair_indices=[0],
        gt_pairs=gt_pairs,
        baseline_seg=0.001,
        baseline_pose=0.0001,
        distortion_net=distortion_net,
        device=torch.device("cpu"),
        rng=None,
        noise_sigmas=[1.0],
    )
    nonzero_deltas = [r for r in results if r.delta_score_components != 0.0]
    assert len(nonzero_deltas) >= 1, (
        f"Expected at least one non-zero Δscore at sigma=1.0; got results: "
        f"{[(r.target, r.delta_score_components) for r in results]}"
    )


def test_pr106_tier_c_explicit_sigma_list_respected():
    """Caller-supplied noise_sigmas list overrides the default."""
    if not CANONICAL_PR106_ARCHIVE.is_file():
        pytest.skip("canonical PR106 archive not present in this checkout")
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes = _read_pr106_inner_bytes()
    assert inner_bytes is not None
    gt_pairs = _build_gt_pairs(1)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(0)
    results = mod._run_tier_c_pr106(
        inner_bytes=inner_bytes,
        pair_indices=[0],
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


def test_pr106_tier_c_empty_sigma_list_returns_empty():
    """Empty sigma list → empty results (skip-tier-c contract)."""
    if not CANONICAL_PR106_ARCHIVE.is_file():
        pytest.skip("canonical PR106 archive not present in this checkout")
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    mod = _load_module()

    inner_bytes = _read_pr106_inner_bytes()
    assert inner_bytes is not None
    gt_pairs = _build_gt_pairs(1)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(0)
    results = mod._run_tier_c_pr106(
        inner_bytes=inner_bytes,
        pair_indices=[0],
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
# Bug-class anti-regression
# ----------------------------------------------------------------------


def test_pr106_tier_c_function_distinct_from_other_grammar_implementations():
    """_run_tier_c_pr106 must be a separate callable from sister Tier C functions."""
    mod = _load_module()
    assert hasattr(mod, "_run_tier_c_pr106"), "PR106 Tier C function missing"
    assert hasattr(mod, "_run_tier_c_a1")
    assert hasattr(mod, "_run_tier_c_ibps1")
    assert hasattr(mod, "_run_tier_c_dp1")
    # All four must be distinct callables (no aliasing).
    assert mod._run_tier_c_pr106 is not mod._run_tier_c_a1
    assert mod._run_tier_c_pr106 is not mod._run_tier_c_ibps1
    assert mod._run_tier_c_pr106 is not mod._run_tier_c_dp1


def test_pr106_grammar_no_longer_falls_through_to_empty():
    """grammar='pr106*' must NOT silently return [] (the pre-c4938a25f bug)."""
    mod = _load_module()
    # Mock _run_tier_c_pr106 to return a sentinel; verify the dispatch
    # actually CALLS it (not silently falling through to the catch-all []).
    sentinel = ["pr106-was-called"]
    with mock.patch.object(mod, "_run_tier_c_pr106", return_value=sentinel):
        for g in ("pr106", "pr106_latent_sidecar", "pr106_latent_sidecar_r2"):
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


def test_pr106_grammar_in_supported_grammars_set():
    """SUPPORTED_GRAMMARS canonical token includes pr106_latent_sidecar."""
    mod = _load_module()
    assert "pr106_latent_sidecar" in mod.SUPPORTED_GRAMMARS


# ----------------------------------------------------------------------
# JSON serialization round-trip
# ----------------------------------------------------------------------


def test_pr106_tier_c_results_json_serializable():
    """TierCResult instances from PR106 Tier C survive JSON round-trip.

    Anti-regression for the empirical pipeline: results land as JSON in
    the ablation output dir; if the schema diverges across grammars,
    downstream aggregation breaks.
    """
    if not CANONICAL_PR106_ARCHIVE.is_file():
        pytest.skip("canonical PR106 archive not present in this checkout")
    torch = pytest.importorskip("torch")
    pytest.importorskip("brotli")
    import dataclasses
    import json
    mod = _load_module()

    inner_bytes = _read_pr106_inner_bytes()
    assert inner_bytes is not None
    gt_pairs = _build_gt_pairs(1)
    distortion_net = _FakeDistortionNet()

    torch.manual_seed(42)
    results = mod._run_tier_c_pr106(
        inner_bytes=inner_bytes,
        pair_indices=[0],
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


def test_pr106_grammar_aliases_normalized_to_canonical():
    """All PR106 grammar aliases normalize to ``pr106_latent_sidecar`` at the
    grammar-normalizer surface.

    Cross-references the parse_pr106_archive_bytes contract used at the
    Tier A surface.
    """
    mod = _load_module()
    assert mod.normalize_grammar("pr106") == "pr106_latent_sidecar"
    assert mod.normalize_grammar("PR106") == "pr106_latent_sidecar"
    assert (
        mod.normalize_grammar("pr106_latent_sidecar_r2")
        == "pr106_latent_sidecar"
    )
    assert (
        mod.normalize_grammar("pr106_latent_sidecar")
        == "pr106_latent_sidecar"
    )


# ----------------------------------------------------------------------
# Real PR106 canonical archive parses correctly under load_archive
# ----------------------------------------------------------------------


def test_load_archive_pr106_parses_canonical_archive():
    """load_archive(canonical PR106 archive, 'pr106*') yields valid sections."""
    if not CANONICAL_PR106_ARCHIVE.is_file():
        pytest.skip("canonical PR106 archive not present in this checkout")
    mod = _load_module()
    inner_bytes, sections = mod.load_archive(
        CANONICAL_PR106_ARCHIVE, "pr106_latent_sidecar"
    )
    assert isinstance(inner_bytes, bytes)
    assert len(inner_bytes) > 0
    # PR106 grammar emits these section names.
    expected = {
        "magic_format_header",
        "pr106_len_field",
        "pr106_base_archive",
        "sidecar_len_field",
        "sidecar_blob",
    }
    assert set(sections.keys()) == expected
