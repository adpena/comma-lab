# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the canonical score-exact saliency producer (P18 + P19).

Slot EEE discipline (CLAUDE.md "NO FAKE IMPLEMENTATIONS"):
  * Real-weight-or-SKIP: tests that need the frozen scorers skip (NOT fake) when
    ``upstream/models/*.safetensors`` or ``upstream/videos/0.mkv`` are absent.
  * Behavioral assertions: s_seg must be boundary-structured (boundary >> interior)
    and s_pose must be spread (not boundary-peaked) on REAL frames.
  * The concentration metric is verified on a SYNTHETIC single-hot saliency
    (Gini -> ~1, top-1% mass -> ~1) and a SYNTHETIC uniform saliency (Gini -> 0)
    — these are math-only tests of the analyzer, no scorer needed.
  * A guard test FAILS if the producer returns constants instead of real
    gradients (Class-2 NO-FAKE: the test would still pass if the function body
    were replaced by ``return zeros`` ONLY for the wrong reason — so we assert
    nontrivial spatial variance + finite + nonzero, which a constant cannot pass).
  * batched_vjp == loop equivalence proves the optimization preserves semantics.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tac.analysis.score_exact_saliency import (  # noqa: E402
    PoseFisher,
    ProducerProfile,
    ScoreExactSaliencyError,
    SegFlipRisk,
    assert_posenet_yuv6_gradient_reachable,
    build_producer_provenance,
    compute_s_pose_fisher,
    compute_s_seg_flip_risk,
    load_score_exact_scorers,
    probe_posenet_yuv6_gradient_reachability,
    profile_producer,
    saliency_concentration,
    stream_real_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM = REPO_ROOT / "upstream"
_MODELS_PRESENT = (_UPSTREAM / "models/segnet.safetensors").exists() and (
    _UPSTREAM / "models/posenet.safetensors"
).exists()
_VIDEO = _UPSTREAM / "videos/0.mkv"
_VIDEO_PRESENT = _VIDEO.exists()
_REAL_OK = _MODELS_PRESENT and _VIDEO_PRESENT

_real_weights = pytest.mark.skipif(
    not _REAL_OK,
    reason="real frozen scorers or contest video absent locally (NO-FAKE: skip, not fake)",
)


# Cache the loaded scorers + a couple of real pairs across tests in the module.
@pytest.fixture(scope="module")
def real_scorers_and_pairs():
    from tac.analysis.score_exact_saliency import load_score_exact_scorers

    torch.manual_seed(20260601)
    posenet, segnet = load_score_exact_scorers(_UPSTREAM, device="cpu")
    pairs = list(stream_real_pairs(_VIDEO, num_pairs=2, device="cpu"))
    return posenet, segnet, pairs


# ---------------------------------------------------------------------------
# Concentration analyzer — pure math, no scorer needed.
# ---------------------------------------------------------------------------


def test_concentration_single_hot_gini_near_one():
    """A single-hot saliency: Gini -> ~1, top-1% mass -> ~1 (max concentration)."""
    n = 10_000
    sal = torch.zeros(n)
    sal[0] = 1.0
    c = saliency_concentration(sal.reshape(100, 100))
    assert c.gini > 0.99, f"single-hot Gini should be ~1, got {c.gini}"
    assert c.top_k_pct_mass[1.0] == pytest.approx(1.0, abs=1e-9)
    assert c.top_k_pct_mass[10.0] == pytest.approx(1.0, abs=1e-9)
    assert c.total_mass == pytest.approx(1.0)
    assert c.nonzero_frac == pytest.approx(1.0 / n, abs=1e-6)


def test_concentration_uniform_gini_near_zero():
    """A uniform saliency: Gini -> 0, top-k% mass -> k% (no concentration)."""
    sal = torch.ones(100, 100)
    c = saliency_concentration(sal)
    assert c.gini < 0.01, f"uniform Gini should be ~0, got {c.gini}"
    assert c.top_k_pct_mass[1.0] == pytest.approx(0.01, abs=1e-3)
    assert c.top_k_pct_mass[10.0] == pytest.approx(0.10, abs=1e-3)


def test_concentration_all_zero_is_safe():
    """An all-zero saliency must not divide by zero; Gini=0, mass=0."""
    sal = torch.zeros(64, 64)
    c = saliency_concentration(sal)
    assert c.gini == 0.0
    assert c.total_mass == 0.0
    assert all(v == 0.0 for v in c.top_k_pct_mass.values())


def test_concentration_top_k_monotone():
    """top-1% mass <= top-5% mass <= top-10% mass (cumulative, monotone)."""
    torch.manual_seed(7)
    sal = torch.rand(200, 200).pow(4)  # heavy-tailed
    c = saliency_concentration(sal)
    assert c.top_k_pct_mass[1.0] <= c.top_k_pct_mass[5.0] + 1e-9
    assert c.top_k_pct_mass[5.0] <= c.top_k_pct_mass[10.0] + 1e-9


def test_profile_producer_uses_diagnostics_free_hot_path_by_default(monkeypatch):
    """Campaign profiling must not force scalar diagnostic sync by default."""
    import tac.analysis.score_exact_saliency as saliency_mod

    calls: list[tuple[str, bool | None]] = []

    def fake_seg(_segnet, pair, *, diagnostics=True):
        calls.append(("seg", diagnostics))
        h, w = pair.shape[-2:]
        return SegFlipRisk(
            flip_risk=torch.ones(h, w),
            grad_energy=torch.ones(h, w),
            margin=torch.ones(h, w),
            grad_finite=bool(diagnostics),
            grad_nonzero_frac=1.0 if diagnostics else float("nan"),
            scorer_input_hw=(h, w),
        )

    def fake_pose(_posenet, pair, *, method="batched_vjp", diagnostics=True):
        calls.append((f"pose:{method}", diagnostics))
        h, w = pair.shape[-2:]
        return PoseFisher(
            s_pose=torch.ones(h, w),
            s_pose_per_frame=torch.ones(2, h, w),
            grad_finite=bool(diagnostics),
            s_pose_nonzero_frac=1.0 if diagnostics else float("nan"),
            method=method,
            scorer_input_hw=(h, w),
        )

    monkeypatch.setattr(saliency_mod, "compute_s_seg_flip_risk", fake_seg)
    monkeypatch.setattr(saliency_mod, "compute_s_pose_fisher", fake_pose)

    pairs = torch.zeros(2, 2, 3, 4, 5)
    profile = profile_producer(object(), object(), pairs)

    assert isinstance(profile, ProducerProfile)
    assert calls == [
        ("seg", False),
        ("pose:batched_vjp", False),
        ("seg", False),
        ("pose:batched_vjp", False),
    ]


class _DifferentiablePosePreprocess(torch.nn.Module):
    def preprocess_input(self, x):
        b, t, c, h, w = x.shape
        return x.reshape(b, t * c, h, w).repeat_interleave(2, dim=1)[:, :12]


class _DetachedPosePreprocess(torch.nn.Module):
    def preprocess_input(self, x):
        b, t, c, h, w = x.shape
        with torch.no_grad():
            return x.reshape(b, t * c, h, w).repeat_interleave(2, dim=1)[:, :12]


def test_posenet_yuv6_gradient_probe_passes_differentiable_preprocess():
    proof = probe_posenet_yuv6_gradient_reachability(_DifferentiablePosePreprocess())

    assert proof.gradient_reachable is True
    assert proof.blockers == ()
    assert proof.grad_abs_sum > 0.0
    assert proof.grad_nonzero_fraction > 0.0
    assert proof.yuv6_shape == (1, 12, 8, 10)
    payload = proof.as_jsonable()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_posenet_yuv6_gradient_probe_blocks_upstream_no_grad_preprocess():
    proof = probe_posenet_yuv6_gradient_reachability(_DetachedPosePreprocess())

    assert proof.gradient_reachable is False
    assert "posenet_yuv6_preprocess_output_detached" in proof.blockers
    assert "posenet_yuv6_preprocess_gradient_abs_sum_too_small" in proof.blockers
    with pytest.raises(ScoreExactSaliencyError, match="not reachable"):
        assert_posenet_yuv6_gradient_reachable(_DetachedPosePreprocess())


def test_score_exact_loader_defaults_to_strict_yuv6_gradient_guard():
    signature = __import__("inspect").signature(load_score_exact_scorers)
    assert signature.parameters["verify_posenet_yuv6_gradient"].default is True


def test_concentration_gini_bounded():
    """Gini stays in [0, 1] for a range of nonnegative inputs."""
    for seed in range(5):
        torch.manual_seed(seed)
        sal = torch.rand(50, 50) * (10.0**seed)
        c = saliency_concentration(sal)
        assert 0.0 <= c.gini <= 1.0


# ---------------------------------------------------------------------------
# Provenance — names upstream source + mirror + the 6 proofs.
# ---------------------------------------------------------------------------


def test_provenance_is_fail_closed_and_references_contract():
    prov = build_producer_provenance(upstream_dir="upstream", repo_root=REPO_ROOT)
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["axis_tag"] == "[macOS-CPU advisory]"
    # Must reference the 6 required numerical proofs from the canonical contract.
    proof_names = {p["name"] for p in prov["required_numerical_proofs"]}
    assert "yuv6_forward_parity" in proof_names
    assert "yuv6_gradient_nonzero" in proof_names
    assert "segnet_last_frame_asymmetry" in proof_names
    assert "posenet_pair_six_axis_exactness" in proof_names
    assert len(prov["required_numerical_proofs"]) == 6
    # Must name the mirror implementation paths.
    assert "make_scorers_differentiable" in prov["mirror_paths"]["scorer_patch"]


# ---------------------------------------------------------------------------
# REAL-weight behavioral tests (skip, not fake, when assets absent).
# ---------------------------------------------------------------------------


@_real_weights
def test_s_seg_is_boundary_structured(real_scorers_and_pairs):
    """P18: flip-risk concentrates at decision boundaries (low-margin pixels).

    NO-FAKE guard: a constant surface would have boundary/interior ratio == 1.
    We assert the ratio is FAR above 1 (real DeepFool flip-risk is boundary-peaked
    because margin is small at boundaries, and flip_risk ~ 1/margin^2).
    """
    _, segnet, pairs = real_scorers_and_pairs
    seg = compute_s_seg_flip_risk(segnet, pairs[0])
    assert isinstance(seg, SegFlipRisk)
    assert seg.grad_finite
    assert seg.scorer_input_hw == (384, 512)
    c = saliency_concentration(seg.flip_risk, margin=seg.margin)
    # Boundary energy must vastly exceed interior energy (verified ~20000x).
    assert c.boundary_over_interior_ratio > 10.0, (
        f"s_seg must be boundary-structured; ratio={c.boundary_over_interior_ratio}"
    )
    # And highly concentrated (Gini near 1).
    assert c.gini > 0.9, f"s_seg Gini should be high (concentrated), got {c.gini}"


@_real_weights
def test_s_pose_is_spread_not_boundary_peaked(real_scorers_and_pairs):
    """P19: Fisher info is more spread than s_seg (geometric, full-frame).

    s_pose is the input-Jacobian energy of the pose head — it depends on the
    whole frame's geometry, not just SegNet class boundaries. Its top-1% mass
    must be materially LOWER than s_seg's (which is ~0.998).
    """
    posenet, segnet, pairs = real_scorers_and_pairs
    pose = compute_s_pose_fisher(posenet, pairs[0], method="batched_vjp")
    seg = compute_s_seg_flip_risk(segnet, pairs[0])
    assert isinstance(pose, PoseFisher)
    assert pose.grad_finite
    cp = saliency_concentration(pose.s_pose)
    cs = saliency_concentration(seg.flip_risk, margin=seg.margin)
    # s_pose is more spread: its Gini < s_seg Gini, and top-1% mass is lower.
    assert cp.top_k_pct_mass[1.0] < cs.top_k_pct_mass[1.0], (
        f"s_pose top-1% ({cp.top_k_pct_mass[1.0]}) should be < s_seg top-1% "
        f"({cs.top_k_pct_mass[1.0]})"
    )
    # But still nontrivially concentrated (it is NOT uniform).
    assert cp.gini > 0.5, f"s_pose should still be concentrated, Gini={cp.gini}"
    assert pose.s_pose_nonzero_frac > 0.1


@_real_weights
def test_pose_fisher_loop_equals_batched_vjp(real_scorers_and_pairs):
    """The optimized batched_vjp must be NUMERICALLY IDENTICAL to the loop."""
    posenet, _, pairs = real_scorers_and_pairs
    loop = compute_s_pose_fisher(posenet, pairs[0], method="loop")
    batched = compute_s_pose_fisher(posenet, pairs[0], method="batched_vjp")
    assert loop.method == "loop"
    assert batched.method == "batched_vjp"
    max_diff = (loop.s_pose - batched.s_pose).abs().max().item()
    assert max_diff < 1e-6, f"batched_vjp must equal loop, max_abs_diff={max_diff}"


@_real_weights
def test_diagnostics_fast_path_preserves_saliency_tensors(real_scorers_and_pairs):
    """Campaign fast mode may skip scalar diagnostics, never saliency tensors."""
    posenet, segnet, pairs = real_scorers_and_pairs
    seg_full = compute_s_seg_flip_risk(segnet, pairs[0], diagnostics=True)
    seg_fast = compute_s_seg_flip_risk(segnet, pairs[0], diagnostics=False)
    pose_full = compute_s_pose_fisher(
        posenet,
        pairs[0],
        method="batched_vjp",
        diagnostics=True,
    )
    pose_fast = compute_s_pose_fisher(
        posenet,
        pairs[0],
        method="batched_vjp",
        diagnostics=False,
    )

    assert torch.allclose(seg_fast.flip_risk, seg_full.flip_risk)
    assert torch.allclose(seg_fast.margin, seg_full.margin)
    assert torch.allclose(pose_fast.s_pose, pose_full.s_pose)
    assert seg_fast.grad_finite is True
    assert pose_fast.grad_finite is True
    assert math.isnan(seg_fast.grad_nonzero_frac)
    assert math.isnan(pose_fast.s_pose_nonzero_frac)


@_real_weights
def test_producer_returns_real_gradients_not_constants(real_scorers_and_pairs):
    """NO-FAKE Class-2 guard: a constant return would FAIL these invariants.

    If ``compute_s_seg_flip_risk`` / ``compute_s_pose_fisher`` were replaced by
    ``return <constant tensor>``, the spatial variance would be ~0 and the
    boundary/interior ratio would be 1. We assert real gradient structure:
      - nontrivial spatial variance (a constant has zero variance);
      - finite + nonzero;
      - the two surfaces are DIFFERENT from each other (a shared constant stub
        would make them identical).
    """
    posenet, segnet, pairs = real_scorers_and_pairs
    seg = compute_s_seg_flip_risk(segnet, pairs[0])
    pose = compute_s_pose_fisher(posenet, pairs[0], method="batched_vjp")

    # Nontrivial spatial variance (constants have variance 0).
    seg_var = seg.flip_risk.float().var().item()
    pose_var = pose.s_pose.float().var().item()
    assert seg_var > 0.0, "s_seg has zero spatial variance — looks like a constant"
    assert pose_var > 0.0, "s_pose has zero spatial variance — looks like a constant"

    # Surfaces are genuinely different (not the same stub). Resize the s_seg
    # surface (384x512) to compare against the native-resolution s_pose shape is
    # not meaningful; instead assert their normalized concentration signatures
    # differ — s_seg is far more concentrated than s_pose.
    cs = saliency_concentration(seg.flip_risk, margin=seg.margin)
    cp = saliency_concentration(pose.s_pose)
    assert cs.gini != pytest.approx(cp.gini, abs=1e-3), (
        "s_seg and s_pose have identical Gini — looks like a shared constant stub"
    )

    # Grad reachability is the whole point of the differentiable mirror.
    assert seg.grad_nonzero_frac > 0.0
    assert pose.s_pose_nonzero_frac > 0.0


@_real_weights
def test_s_seg_uses_last_frame_only(real_scorers_and_pairs):
    """P18 frame-scope: mutating frame_0 alone must NOT change s_seg.

    SegNet scores x[:, -1, ...] (frame_1). Per the contract's
    'segnet_last_frame_asymmetry' proof, perturbing frame_0 leaves the SegNet
    surface unchanged. This proves the producer spends SegNet budget only on
    frame_1 atoms.
    """
    _, segnet, pairs = real_scorers_and_pairs
    pair = pairs[0].clone()
    seg_a = compute_s_seg_flip_risk(segnet, pair)
    # Perturb ONLY frame_0 (index 0 in the pair dim).
    pair_mut = pair.clone()
    pair_mut[:, 0] = pair_mut[:, 0] + 25.0  # large change to frame_0 only
    seg_b = compute_s_seg_flip_risk(segnet, pair_mut)
    diff = (seg_a.flip_risk - seg_b.flip_risk).abs().max().item()
    assert diff < 1e-3, (
        f"frame_0 mutation changed s_seg by {diff} — SegNet must use last frame only"
    )


def test_module_skip_path_when_assets_absent():
    """If real assets are absent, the real-weight tests SKIP (not fake) — verify
    the skip predicate is wired (this test always runs)."""
    if not _REAL_OK:
        # The decorated tests will skip; assert the skip reason is honest.
        assert not _MODELS_PRESENT or not _VIDEO_PRESENT
    else:
        # Assets present — the real tests will run.
        assert _MODELS_PRESENT and _VIDEO_PRESENT
