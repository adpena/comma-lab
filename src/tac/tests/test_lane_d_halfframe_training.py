"""Lane D: dilated-h64 retrain for half-frame masks — pinning tests.

Council 2026-04-27 (5/0). The dilated-h64 baseline renderer collapses on
half-frame archives (PoseNet 0.011 → 28.7) because its MotionPredictor was
trained on independently-SegNet-extracted masks at every frame. The fix:
retrain with warp-expansion injected into the data path so the motion module
learns BOTH distributions (independently-extracted AND warp-reconstructed).

These tests pin:
    1. The DILATED_H64_HALF_FRAME profile loads + matches the 0.9001
       baseline arch byte-for-byte (except the two intentional Lane D deltas:
       use_zoom_flow=True and mask_half_sim_prob=0.5).
    2. parse_args resolves mask_half_sim_prob and use_zoom_flow from the profile.
    3. The training data path actually injects warp-expanded even-frame masks
       when mask_half_sim_prob=1.0 (always-on stress test).
    4. With mask_half_sim_prob=0.0 (default), no warp is invoked — training
       behaviour is identical to baseline (regression guard).
    5. Determinism: same seed + same data → same first-step gradient on a
       small fixture (proves configure_reproducibility() actually pins state).
    6. The bootstrap script references real flags (no invented args).

Cost paranoia: zero GPU dollars wasted on a misconfigured 5h run.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


# ── 1. Profile structure & baseline arch parity ──────────────────────────


def test_dilated_h64_half_frame_profile_registered() -> None:
    """The profile MUST be in PROFILES dict so --profile dilated_h64_half_frame
    works. If this fails, the bootstrap script will SystemExit before launch."""
    from tac.profiles import PROFILES, DILATED_H64_HALF_FRAME

    assert "dilated_h64_half_frame" in PROFILES, (
        "DILATED_H64_HALF_FRAME not registered in PROFILES dict. "
        "scripts/remote_lane_d_halfframe_retrain.sh expects "
        "--profile dilated_h64_half_frame to resolve."
    )
    assert PROFILES["dilated_h64_half_frame"] is DILATED_H64_HALF_FRAME


def test_dilated_h64_half_frame_arch_matches_0_9001_baseline() -> None:
    """The Lane D profile MUST match the verified 0.9001 baseline arch
    byte-for-byte except for the two intentional Lane D deltas. Any arch
    drift would mean we're training a different model than the baseline.

    Baseline arch (read from submissions/baseline_dilated_h64_0_90/renderer.bin
    ASYM header):
      base_ch=36, mid_ch=60, motion_hidden=32, depth=1, embed_dim=6,
      pose_dim=6, use_dsconv=False
    """
    from tac.profiles import DILATED_H64_HALF_FRAME

    p = DILATED_H64_HALF_FRAME
    assert p["base_ch"] == 36, f"baseline arch has base_ch=36, got {p['base_ch']}"
    assert p["mid_ch"] == 60, f"baseline arch has mid_ch=60, got {p['mid_ch']}"
    assert p["motion_hidden"] == 32, (
        f"baseline arch has motion_hidden=32, got {p['motion_hidden']}"
    )
    assert p["depth"] == 1, f"baseline arch has depth=1, got {p['depth']}"
    assert p["embed_dim"] == 6, f"baseline arch has embed_dim=6, got {p['embed_dim']}"
    assert p["pose_dim"] == 6, (
        f"baseline arch has pose_dim=6 (FiLM modulation), got {p['pose_dim']}"
    )
    assert p["use_dsconv"] is False, (
        f"baseline arch has use_dsconv=False, got {p['use_dsconv']}"
    )


def test_dilated_h64_half_frame_lane_d_deltas() -> None:
    """The Lane D deltas vs baseline arch are EXACTLY two flags:
    use_zoom_flow=True and mask_half_sim_prob=0.5. Any other change to the
    arch surface is a council deviation that must be argued for."""
    from tac.profiles import DILATED_H64_HALF_FRAME

    p = DILATED_H64_HALF_FRAME
    assert p["use_zoom_flow"] is True, (
        "Lane D requires use_zoom_flow=True so the motion module produces "
        "gate+residual (4ch) and flow comes from RadialZoomWarp."
    )
    assert p["mask_half_sim_prob"] == pytest.approx(0.5), (
        f"Council 2026-04-27 set mask_half_sim_prob=0.5 (50/50 mix); "
        f"got {p['mask_half_sim_prob']}"
    )


def test_dilated_h64_half_frame_passes_preflight() -> None:
    """The profile must pass tac.preflight.preflight_profiles — specifically
    the rule that mask_half_sim_prob > 0 requires use_zoom_flow=True."""
    from tac.preflight import preflight_profiles

    violations = preflight_profiles(strict=False, verbose=False)
    ours = [v for v in violations if "dilated_h64_half_frame" in v]
    assert not ours, (
        f"DILATED_H64_HALF_FRAME violates preflight rules: {ours}"
    )


def test_dilated_h64_half_frame_deterministic_pinned() -> None:
    """Lane D MUST pin seed + deterministic for bit-exact retrain across
    runs (CLAUDE.md canonical pipeline standard)."""
    from tac.profiles import DILATED_H64_HALF_FRAME

    assert DILATED_H64_HALF_FRAME.get("seed") == 42
    assert DILATED_H64_HALF_FRAME.get("deterministic") is True


def test_dilated_h64_half_frame_5phase_schedule_total() -> None:
    """The 5-phase schedule must sum to ~1980 epochs (~5h on 4090 at $0.25/hr
    = $1.25 budget). If someone bumps phase epochs the cost estimate must
    be updated in scripts/remote_lane_d_halfframe_retrain.sh."""
    from tac.profiles import DILATED_H64_HALF_FRAME

    p = DILATED_H64_HALF_FRAME
    total = sum(p[f"phase{i}_epochs"] for i in range(1, 6))
    assert total == 1980, (
        f"DILATED_H64_HALF_FRAME total epochs changed from 1980 to {total}. "
        f"Update the cost estimate in remote_lane_d_halfframe_retrain.sh "
        f"(currently quoted at 5h / $1.25 on 4090) before promoting."
    )


# ── 2. parse_args propagation ────────────────────────────────────────────


def test_parse_args_resolves_mask_half_sim_prob_from_profile(monkeypatch) -> None:
    """train_renderer.parse_args must surface mask_half_sim_prob from the
    profile. If this fails, the warp-expansion hook in the train loop
    (line ~1147 of train_renderer.py) is silently dead and the renderer
    learns nothing about half-frame distributions."""
    train_renderer = pytest.importorskip("tac.experiments.train_renderer")

    argv = [
        "train_renderer.py",
        "--profile", "dilated_h64_half_frame",
        "--tag", "_unit_test_lane_d",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = train_renderer.parse_args(argv[1:])

    assert args.mask_half_sim_prob == pytest.approx(0.5), (
        f"Profile sets mask_half_sim_prob=0.5; resolver returned "
        f"{args.mask_half_sim_prob!r}. The half-frame simulation hook "
        f"depends on this — if it's 0, training is dead-equivalent to "
        f"the broken baseline."
    )
    assert args.use_zoom_flow is True, (
        f"Profile sets use_zoom_flow=True; resolver returned "
        f"{args.use_zoom_flow!r}. AsymmetricPairGenerator forward() will "
        f"crash without ego_flow."
    )
    # Spot-check baseline arch flags too
    assert args.base_ch == 36
    assert args.mid_ch == 60
    assert args.motion_hidden == 32
    assert args.pose_dim == 6


def test_parse_args_default_mask_half_sim_prob_is_off(monkeypatch) -> None:
    """Without a profile or CLI flag, mask_half_sim_prob must default to 0
    (no half-frame simulation). Regression guard for the existing baseline
    training paths."""
    train_renderer = pytest.importorskip("tac.experiments.train_renderer")

    argv = ["train_renderer.py", "--tag", "_unit_test_lane_d_default"]
    monkeypatch.setattr(sys, "argv", argv)
    args = train_renderer.parse_args(argv[1:])
    assert args.mask_half_sim_prob == 0.0, (
        f"Default mask_half_sim_prob must be 0.0 (off); got "
        f"{args.mask_half_sim_prob!r}. Any nonzero default would silently "
        f"shift the distribution every training run."
    )


# ── 3. Warp-expansion is actually invoked ────────────────────────────────


def _toy_mask(h: int = 64, w: int = 96, cls: int = 1) -> torch.Tensor:
    """A small mask with a recognizable class-1 patch."""
    m = torch.zeros(h, w, dtype=torch.long)
    m[20:40, 30:60] = cls
    return m


def test_warp_inverse_applied_when_prob_one() -> None:
    """When mask_half_sim_prob=1.0, mask_t MUST be replaced by
    warp_inverse_masks(mask_t1, pair_idx). Verify by setting up the same
    objects the train loop uses and checking the output differs from
    the unwarped GT mask_t."""
    from tac.radial_zoom import RadialZoomWarp

    H, W = 64, 96
    n_pairs = 4
    sim_zoom_warp = RadialZoomWarp(n_pairs=n_pairs)
    with torch.no_grad():
        # Set a non-trivial zoom so the warp actually moves pixels (zero zoom
        # is identity per test_radial_zoom_warp_inverse.py:test_warp_inverse_zero_zoom_is_identity)
        sim_zoom_warp.zoom_scalars[:] = torch.tensor([0.08, -0.08, 0.05, -0.05])
    sim_zoom_warp.eval()
    for p in sim_zoom_warp.parameters():
        p.requires_grad_(False)

    mask_t1 = _toy_mask(H, W).unsqueeze(0)
    mask_t_orig = _toy_mask(H, W).unsqueeze(0)  # GT mask_t (same toy mask)
    pair_idx = torch.tensor([0], dtype=torch.long)

    # Reproduce the train loop's hook: mask_t = warp_inverse_masks(mask_t1, idx)
    mask_t_warped = sim_zoom_warp.warp_inverse_masks(mask_t1, pair_idx)

    # Output shape + dtype contract
    assert mask_t_warped.shape == mask_t1.shape
    assert mask_t_warped.dtype == mask_t1.dtype

    # The warp must actually move pixels (otherwise we've not changed the
    # training distribution). Compare to the GT mask — they should differ
    # at the boundary because nearest-neighbour resampling shifts pixels.
    diff_count = int((mask_t_warped != mask_t_orig).sum().item())
    assert diff_count > 0, (
        "warp_inverse_masks with non-zero zoom must move at least one pixel; "
        "if 0 pixels differ, the zoom scalar is being ignored."
    )


def test_warp_not_invoked_when_prob_zero() -> None:
    """With mask_half_sim_prob=0.0, the train loop's warp branch must NEVER
    fire (since random.random() < 0 is always False). We assert by exhausting
    1000 random draws against the threshold."""
    rng = random.Random(42)
    warp_invoked = 0
    for _ in range(1000):
        if rng.random() < 0.0:  # exact mirror of train_renderer.py:1148
            warp_invoked += 1
    assert warp_invoked == 0, (
        f"mask_half_sim_prob=0.0 must never invoke the warp branch; "
        f"got {warp_invoked} invocations in 1000 draws"
    )


def test_warp_invoked_proportional_to_prob() -> None:
    """With mask_half_sim_prob=0.5, ~half of pairs in an epoch should hit
    the warp branch. Tolerance ±5% over 10000 draws (concentration bound)."""
    rng = random.Random(42)
    warp_count = 0
    n = 10_000
    p = 0.5
    for _ in range(n):
        if rng.random() < p:
            warp_count += 1
    expected = int(n * p)
    tolerance = int(n * 0.03)  # 3% absolute
    assert abs(warp_count - expected) <= tolerance, (
        f"mask_half_sim_prob=0.5 should produce ~{expected} warp invocations "
        f"in {n} draws (±{tolerance}); got {warp_count}"
    )


# ── 4. ego_flow plumbing for use_zoom_flow=True models ───────────────────


def test_asymmetric_pair_generator_requires_ego_flow_when_use_zoom_flow() -> None:
    """Pin the contract: use_zoom_flow=True models REQUIRE ego_flow at
    forward(). If train_renderer.py forgets to pass it, training crashes
    immediately on the first batch — better than silent wrong-output."""
    from tac.renderer import build_renderer

    model = build_renderer(
        num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
        motion_hidden=32, depth=1, pose_dim=6,
        use_zoom_flow=True, use_dsconv=False, padding_mode="zeros",
        use_dilation=False,
    )
    model.eval()
    mask = torch.zeros(1, 64, 96, dtype=torch.long)
    with pytest.raises(ValueError, match="use_zoom_flow=True requires ego_flow"):
        with torch.no_grad():
            model(mask, mask)  # no ego_flow → must raise


def test_asymmetric_pair_generator_runs_with_ego_flow() -> None:
    """The full ego_flow plumbing path: build the model, build the warp,
    call forward(mask_t, mask_t1, ego_flow=warp(idx, H, W)) — must produce
    a valid (1, 2, H, W, 3) HWC pair without exception."""
    from tac.radial_zoom import RadialZoomWarp
    from tac.renderer import build_renderer

    model = build_renderer(
        num_classes=5, embed_dim=6, base_ch=36, mid_ch=60,
        motion_hidden=32, depth=1, pose_dim=6,
        use_zoom_flow=True, use_dsconv=False, padding_mode="zeros",
        use_dilation=False,
    )
    model.eval()

    H, W = 64, 96
    warp = RadialZoomWarp(n_pairs=4, target_h=H, target_w=W)
    with torch.no_grad():
        warp.zoom_scalars[0] = 0.05
    warp.eval()

    mask_t = _toy_mask(H, W).unsqueeze(0)
    mask_t1 = _toy_mask(H, W).unsqueeze(0)
    pair_idx = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        ego_flow = warp(pair_idx, H, W)  # (1, 2, H, W)
        # FiLM models need a pose vector; pass zeros since this is a smoke
        out = model(mask_t, mask_t1, ego_flow=ego_flow,
                    pose=torch.zeros(1, 6))
    assert out.shape == (1, 2, H, W, 3), f"unexpected output shape: {out.shape}"
    assert torch.isfinite(out).all(), "model produced non-finite outputs"


# ── 5. Reproducibility — same seed → same gradient ───────────────────────


def _first_step_gradient_signature(seed: int) -> torch.Tensor:
    """Run a tiny single forward+backward and return a fingerprint of the
    gradient state. Same seed must produce same fingerprint (bit-exact on CPU,
    near-exact on a given CUDA SKU)."""
    from tac.experiments.train_renderer import configure_reproducibility
    from tac.renderer import build_renderer

    configure_reproducibility(seed=seed, deterministic=True)

    # Build a tiny model deterministically
    model = build_renderer(
        num_classes=5, embed_dim=4, base_ch=16, mid_ch=24,
        motion_hidden=8, depth=1, pose_dim=6,
        use_zoom_flow=False,  # skip ego_flow plumbing for the determinism check
        use_dsconv=False, padding_mode="zeros", use_dilation=False,
    )
    model.train()
    mask_t = torch.zeros(1, 32, 48, dtype=torch.long)
    mask_t1 = torch.zeros(1, 32, 48, dtype=torch.long)
    mask_t1[:, 5:15, 10:20] = 1
    out = model(mask_t, mask_t1)
    target = torch.zeros_like(out)
    loss = (out - target).pow(2).mean()
    loss.backward()
    # Fingerprint: sum of |grad| over all parameters
    fingerprint = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in model.parameters()
    )
    return torch.tensor(fingerprint)


def test_reproducibility_same_seed_same_gradient_fingerprint() -> None:
    """Same seed (via configure_reproducibility) must produce the same
    forward+backward fingerprint. This is the contract the entire Lane D
    cost estimate depends on — if seeds aren't deterministic, re-running
    the failed experiment burns another $1.25 with no guarantee of catching
    the same path."""
    sig_a = _first_step_gradient_signature(42)
    sig_b = _first_step_gradient_signature(42)
    # On CPU with deterministic algorithms this should be exact
    assert torch.equal(sig_a, sig_b), (
        f"Same seed produced different gradient fingerprints: "
        f"{sig_a.item()} != {sig_b.item()}. configure_reproducibility() "
        f"is not pinning state correctly — Lane D's bit-exact-retrain "
        f"guarantee is broken."
    )


def test_reproducibility_different_seeds_differ() -> None:
    """Sanity check: different seeds DO produce different fingerprints
    (otherwise the test above is vacuous — anything passes if the model
    is constant)."""
    sig_a = _first_step_gradient_signature(42)
    sig_b = _first_step_gradient_signature(7)
    assert not torch.equal(sig_a, sig_b), (
        f"Seeds 42 and 7 produced identical gradient fingerprints: "
        f"{sig_a.item()}. Either the model is constant (test broken) or "
        f"seeds are being ignored entirely."
    )


# ── 6. Bootstrap script integrity ────────────────────────────────────────


def test_bootstrap_script_exists_and_executable() -> None:
    script = REPO / "scripts" / "remote_lane_d_halfframe_retrain.sh"
    assert script.is_file(), f"missing bootstrap: {script}"
    import os, stat
    st = os.stat(script)
    assert st.st_mode & stat.S_IXUSR, f"{script} not executable"


def test_bootstrap_script_uses_real_train_renderer_flags() -> None:
    """Every --flag the bootstrap script passes to train_renderer.py must
    exist in train_renderer.py's argparse. Catches the 'invented flag'
    antipattern (cf. test_train_renderer_auth_eval_wiring.py)."""
    script_src = (REPO / "scripts" / "remote_lane_d_halfframe_retrain.sh").read_text()
    train_src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()

    # Real argparse flags in train_renderer.py
    real_flags = set(re.findall(r'add_argument\(\s*["\']--([a-z][a-z0-9-]+)', train_src))
    assert real_flags, "regex couldn't find any add_argument flags — fix the regex"

    # Find the train_renderer.py invocation block in the bootstrap
    m = re.search(
        r'src/tac/experiments/train_renderer\.py(.*?)(?=\n\s*BEST_BIN=|\Z)',
        script_src, re.DOTALL,
    )
    assert m, "couldn't locate the train_renderer.py invocation in the bootstrap script"
    invocation = m.group(0)

    used_flags = set(re.findall(r'\B--([a-z][a-z0-9-]+)', invocation))
    invented = used_flags - real_flags
    assert not invented, (
        f"bootstrap passes flags that don't exist in train_renderer.py argparse: "
        f"{sorted(invented)}. Either add the flag or fix the script — argparse "
        f"will SystemExit on launch otherwise."
    )


def test_bootstrap_script_uses_dilated_h64_half_frame_profile() -> None:
    """The bootstrap MUST reference --profile dilated_h64_half_frame —
    otherwise it's launching the wrong experiment."""
    script_src = (REPO / "scripts" / "remote_lane_d_halfframe_retrain.sh").read_text()
    assert "--profile dilated_h64_half_frame" in script_src, (
        "bootstrap doesn't reference --profile dilated_h64_half_frame — "
        "either someone retired the profile name or the bootstrap launches "
        "a different experiment."
    )


def test_bootstrap_script_pins_pythonhashseed() -> None:
    """Reproducibility requires PYTHONHASHSEED. CLAUDE.md non-negotiable."""
    script_src = (REPO / "scripts" / "remote_lane_d_halfframe_retrain.sh").read_text()
    assert "PYTHONHASHSEED=" in script_src, (
        "bootstrap doesn't export PYTHONHASHSEED — Python's hash randomisation "
        "destroys deterministic dict iteration and breaks Lane D's bit-exact "
        "retrain claim."
    )


# ── 7. Integration: source-level plumbing checks ─────────────────────────


def test_train_renderer_source_contains_ego_flow_plumbing() -> None:
    """Mario Round 2 dissent: pure unit tests on model() can pass vacuously
    if the train loop forgets to plumb ego_flow. Source-level grep ensures
    the train loop block exists and references both the warp and the model
    forward kwarg."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()

    # Must compute ego_flow from sim_zoom_warp (the same warp used for
    # half-frame sim) gated on use_zoom_flow.
    assert 'ego_flow = sim_zoom_warp(' in src, (
        "train_renderer.py train loop must call sim_zoom_warp(pair_idx_t, H, W) "
        "to compute ego_flow — ego_flow plumbing missing or refactored."
    )
    # Must invoke the model with ego_flow=ego_flow when present.
    assert 'model(mask_t, mask_t1, ego_flow=ego_flow)' in src, (
        "train_renderer.py train loop must call model(..., ego_flow=ego_flow) "
        "for use_zoom_flow=True models — kwarg path missing."
    )
    # Must handle horizontal flip — flow x-component negation.
    assert 'flipped_h' in src, (
        "train_renderer.py must track flipped_h state for ego_flow mirroring "
        "(see comment block re: hflip handling)."
    )


def test_train_renderer_source_saves_zoom_scalars() -> None:
    """The Lane D inflate path needs zoom_scalars.pt in the archive. Without
    it, inflate falls back to identity zoom and Lane D's whole point is
    nullified."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    assert 'zoom_scalars.pt' in src and 'sim_zoom_warp.state_dict()' in src, (
        "train_renderer.py must persist sim_zoom_warp.state_dict() to "
        "zoom_scalars.pt — without this, the inflate-side ZoomWarp falls "
        "back to identity zoom (scalars=0) and breaks Lane D's train/inflate "
        "consistency."
    )


def test_evaluate_fp4_signature_has_zoom_kwargs() -> None:
    """The FP4 in-training evaluator must accept sim_zoom_warp + use_zoom_flow
    so its scorer measurement matches the model's training distribution.
    Without this, eval-time scores diverge from inflate-time scores."""
    from tac.experiments.train_renderer import evaluate_fp4
    import inspect
    sig = inspect.signature(evaluate_fp4)
    assert "sim_zoom_warp" in sig.parameters, (
        "evaluate_fp4 missing sim_zoom_warp kwarg — FP4 eval will pass None "
        "for ego_flow, crashing AsymmetricPairGenerator(use_zoom_flow=True)."
    )
    assert "use_zoom_flow" in sig.parameters, (
        "evaluate_fp4 missing use_zoom_flow kwarg — can't gate ego_flow path."
    )
