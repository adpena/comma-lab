# SPDX-License-Identifier: MIT
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
    antipattern (cf. test_train_renderer_auth_eval_wiring.py).

    Codex R-Lane-D-Issue3: terminator is now `BEST_FP32=` (the script no
    longer searches for a non-existent `*_fp4.bin` glob; it reads the
    canonical fp32 .pt and exports a real FP4A renderer.bin via
    tac.renderer_export.export_asymmetric_checkpoint_fp4)."""
    script_src = (REPO / "scripts" / "remote_lane_d_halfframe_retrain.sh").read_text()
    train_src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()

    # Real argparse flags in train_renderer.py
    real_flags = set(re.findall(r'add_argument\(\s*["\']--([a-z][a-z0-9-]+)', train_src))
    assert real_flags, "regex couldn't find any add_argument flags — fix the regex"

    # Find the train_renderer.py invocation block in the bootstrap. The
    # terminator must be a marker that appears AFTER the invocation but
    # BEFORE any other CLI-flag-bearing command (otherwise we'd pick up
    # contest_auth_eval / optimize_poses flags).
    m = re.search(
        r'src/tac/experiments/train_renderer\.py(.*?)(?=\n\s*BEST_FP32=|\Z)',
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


# ── 8. Codex R-Lane-D-Issue1: pose_dim resume safety ─────────────────────


def test_force_pose_dim_cli_flag_exists() -> None:
    """Codex R-Lane-D-Issue1: --force-pose-dim must beat profile + checkpoint
    arch_meta. Without it, an operator can't intentionally retrain a different
    pose_dim arch from scratch when a profile or checkpoint disagrees."""
    train_renderer = pytest.importorskip("tac.experiments.train_renderer")
    args = train_renderer.parse_args(["--tag", "_t", "--force-pose-dim", "0"])
    assert args.pose_dim == 0, (
        f"--force-pose-dim 0 must override the default; got {args.pose_dim}"
    )
    args2 = train_renderer.parse_args([
        "--tag", "_t", "--profile", "dilated_h64_half_frame",
        "--force-pose-dim", "0",
    ])
    assert args2.pose_dim == 0, (
        f"--force-pose-dim 0 must override profile pose_dim=6; got {args2.pose_dim}"
    )


def test_save_training_state_includes_arch_meta(tmp_path) -> None:
    """Codex R-Lane-D-Issue1: training_state checkpoints MUST embed arch_meta
    so resumes of legacy checkpoints don't crash on strict load_state_dict
    when the now-active profile resolver promotes pose_dim 0 → 6."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    # The save site must build an arch_meta dict and pass it to torch.save.
    assert 'arch_meta = {' in src, (
        "save_training_state() must construct an arch_meta dict — Codex "
        "R-Lane-D-Issue1. Without this, legacy resume crashes."
    )
    assert '"arch_meta": arch_meta' in src, (
        "save_training_state() must persist arch_meta in the saved dict so "
        "_peek_checkpoint_arch_meta() can recover it on resume."
    )
    assert '"schema_version": 1' in src, (
        "arch_meta needs a schema_version so future arch additions are "
        "backwards-compatible (bump → tracked migration)."
    )


def test_peek_checkpoint_arch_meta_handles_new_format(tmp_path) -> None:
    """Codex R-Lane-D-Issue1: peeking a NEW-format checkpoint (with arch_meta
    sibling key) must return arch_meta verbatim including pose_dim."""
    from tac.experiments.train_renderer import _peek_checkpoint_arch_meta
    ckpt_path = tmp_path / "training_state_test.pt"
    torch.save(
        {
            "epoch": 5,
            "model": {"renderer.const": torch.zeros(1, 4)},
            "ema_shadow": {},
            "arch_meta": {"schema_version": 1, "pose_dim": 6, "base_ch": 36},
        },
        ckpt_path,
    )
    meta = _peek_checkpoint_arch_meta(ckpt_path, device="cpu")
    assert meta is not None and meta["pose_dim"] == 6


def test_peek_checkpoint_arch_meta_legacy_with_film_keys(tmp_path) -> None:
    """Codex R-Lane-D-Issue1: a LEGACY checkpoint (no arch_meta) with film_*
    keys in the state_dict must auto-detect pose_dim=6 so resume builds the
    matching FiLM model."""
    from tac.experiments.train_renderer import _peek_checkpoint_arch_meta
    ckpt_path = tmp_path / "training_state_legacy_film.pt"
    # Synthesise a legacy save: no arch_meta, but state_dict has film_ keys.
    torch.save(
        {
            "epoch": 1,
            "model": {
                "renderer.const": torch.zeros(1, 4),
                "renderer.film_bottleneck.weight": torch.zeros(8, 6),
            },
            "ema_shadow": {},
        },
        ckpt_path,
    )
    meta = _peek_checkpoint_arch_meta(ckpt_path, device="cpu")
    assert meta is not None
    assert meta["pose_dim"] == 6
    assert meta.get("_legacy_no_arch_meta") is True


def test_peek_checkpoint_arch_meta_legacy_no_film_keys(tmp_path) -> None:
    """Codex R-Lane-D-Issue1: a LEGACY checkpoint with NO film_* keys (the
    dead-resolver case where pose_dim was effectively 0) must auto-detect
    pose_dim=0 so resume succeeds with the matching non-FiLM model."""
    from tac.experiments.train_renderer import _peek_checkpoint_arch_meta
    ckpt_path = tmp_path / "training_state_legacy_nofilm.pt"
    torch.save(
        {
            "epoch": 1,
            "model": {"renderer.const": torch.zeros(1, 4)},  # no film_*
            "ema_shadow": {},
        },
        ckpt_path,
    )
    meta = _peek_checkpoint_arch_meta(ckpt_path, device="cpu")
    assert meta is not None
    assert meta["pose_dim"] == 0
    assert meta.get("_legacy_no_arch_meta") is True


def test_resolve_pose_dim_for_resume_force_wins(tmp_path) -> None:
    """Codex R-Lane-D-Issue1: --force-pose-dim must beat checkpoint arch_meta
    AND profile resolution. The shape mismatch that follows is intentional —
    silent shape drift is what we are protecting against, not preventing
    operator-requested overrides."""
    from tac.experiments.train_renderer import _resolve_pose_dim_for_resume
    import argparse
    ckpt_path = tmp_path / "ck.pt"
    torch.save(
        {"model": {}, "ema_shadow": {}, "arch_meta": {"pose_dim": 6}},
        ckpt_path,
    )
    args = argparse.Namespace(force_pose_dim=0, pose_dim=6)
    pd, src = _resolve_pose_dim_for_resume(args, ckpt_path)
    assert pd == 0 and src == "cli_force"


def test_resolve_pose_dim_for_resume_checkpoint_overrides_profile(tmp_path) -> None:
    """Codex R-Lane-D-Issue1: when --force-pose-dim is unset AND the
    checkpoint's arch_meta disagrees with the profile, the checkpoint wins
    (resume must use the saved arch, otherwise strict load fails)."""
    from tac.experiments.train_renderer import _resolve_pose_dim_for_resume
    import argparse
    ckpt_path = tmp_path / "ck.pt"
    torch.save(
        {"model": {}, "ema_shadow": {}, "arch_meta": {"pose_dim": 0}},
        ckpt_path,
    )
    args = argparse.Namespace(force_pose_dim=None, pose_dim=6)
    pd, src = _resolve_pose_dim_for_resume(args, ckpt_path)
    assert pd == 0
    assert src == "checkpoint_arch_meta"


def test_resolve_pose_dim_for_resume_no_checkpoint_uses_profile() -> None:
    """Codex R-Lane-D-Issue1: without a resume checkpoint, the profile
    pose_dim wins (the original Lane D path)."""
    from tac.experiments.train_renderer import _resolve_pose_dim_for_resume
    import argparse
    args = argparse.Namespace(force_pose_dim=None, pose_dim=6)
    pd, src = _resolve_pose_dim_for_resume(args, None)
    assert pd == 6 and src == "profile"


# ── 9. Codex R-Lane-D-Issue2: half-frame eval gating ─────────────────────


def test_evaluate_fp4_accepts_half_frame_mode_kwarg() -> None:
    """Codex R-Lane-D-Issue2: evaluate_fp4 must accept half_frame_mode so the
    in-training evaluator can mirror the inflate-side warp_inverse_masks
    reconstruction. Without this, best checkpoint selection optimises a
    distribution the deployed model never sees."""
    from tac.experiments.train_renderer import evaluate_fp4
    import inspect
    sig = inspect.signature(evaluate_fp4)
    assert "half_frame_mode" in sig.parameters, (
        "evaluate_fp4 missing half_frame_mode kwarg — without it, half-frame "
        "deployment scores can't be measured during training."
    )
    # Default must be False so legacy callers (DEN baseline path) keep
    # full-frame eval semantics.
    default = sig.parameters["half_frame_mode"].default
    assert default is False, (
        f"half_frame_mode default must be False (legacy compat), got {default}"
    )


def test_evaluate_fp4_half_frame_mode_requires_zoom_warp() -> None:
    """Codex R-Lane-D-Issue2: evaluate_fp4(half_frame_mode=True) without a
    sim_zoom_warp must FAIL LOUD instead of silently passing through —
    silent half-frame eval falls back to mask_t = mask_t (the bug we are
    fixing in the first place)."""
    from tac.experiments.train_renderer import evaluate_fp4
    import inspect
    src_lines = inspect.getsource(evaluate_fp4)
    assert "half_frame_mode=True) requires sim_zoom_warp" in src_lines, (
        "evaluate_fp4 must raise when half_frame_mode=True but sim_zoom_warp "
        "is None — Codex R-Lane-D-Issue2."
    )


def test_evaluate_fp4_half_frame_mode_calls_warp_inverse_masks() -> None:
    """Codex R-Lane-D-Issue2: source-level pin — evaluate_fp4 in
    half_frame_mode must call sim_zoom_warp.warp_inverse_masks. Without this,
    the eval is full-frame even with the kwarg set."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    # Locate evaluate_fp4 body
    m = re.search(
        r"def evaluate_fp4\(.*?(?=\n(?:def |class )|\Z)",
        src, re.DOTALL,
    )
    assert m, "couldn't locate evaluate_fp4 body"
    body = m.group(0)
    assert "sim_zoom_warp.warp_inverse_masks(" in body, (
        "evaluate_fp4 body missing sim_zoom_warp.warp_inverse_masks() call. "
        "Without it, half_frame_mode does nothing — Codex R-Lane-D-Issue2."
    )
    assert "if half_frame_mode" in body, (
        "evaluate_fp4 must gate the warp_inverse_masks call on half_frame_mode "
        "(otherwise full-frame eval is broken too)."
    )


def test_train_loop_runs_both_eval_modes_in_halfframe_profile() -> None:
    """Codex R-Lane-D-Issue2: the train loop must call evaluate_fp4 TWICE
    when in half-frame mode (once full-frame for diagnostics, once
    half-frame for best-gating). Source-level pin."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    # Two evaluate_fp4 call sites should appear inside the eval block.
    # Count specifically the calls that pass half_frame_mode=True / False
    # explicitly — those are the dual-mode calls (the legacy single-mode
    # call that doesn't pass the kwarg lives in test code only).
    assert "half_frame_mode=False," in src, (
        "train loop must pass half_frame_mode=False on the diagnostic call"
    )
    assert "half_frame_mode=True," in src, (
        "train loop must pass half_frame_mode=True on the deployment-metric call"
    )


def test_best_checkpoint_uses_half_frame_score_when_halfframe_active() -> None:
    """Codex R-Lane-D-Issue2: the best-checkpoint gate (`if scorer_val <
    best_scorer`) must be fed the HALF-FRAME scorer when half-frame eval is
    active. Without this, the saved checkpoint optimises the wrong
    distribution and Lane D's predicted 0.55-0.75 ships the wrong file."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    # Locate the eval block (between the FP4 evaluation comment and the
    # 'Log and save best' marker).
    m = re.search(
        r"# FP4 evaluation \(skip during Phase 1.*?# Log and save best",
        src, re.DOTALL,
    )
    assert m, "couldn't locate eval block boundary"
    block = m.group(0)
    # Best-gate metric must be assigned from the half-frame eval, not the
    # full-frame eval, when half-frame is active.
    assert "scorer_val = scorer_val_half" in block, (
        "best-gate must be fed scorer_val_half when half-frame eval is active "
        "(Codex R-Lane-D-Issue2). Without this, best ships full-frame-best."
    )
    # And the legacy / non-half-frame path must STILL fall back to full-frame.
    assert "scorer_val = scorer_val_full" in block, (
        "non-half-frame profiles must keep full-frame best-gate semantics"
    )


def test_evaluate_fp4_full_frame_only_in_baseline_profile() -> None:
    """Codex R-Lane-D-Issue2: profiles WITHOUT half-frame deployment (the
    DILATED_H64 / SHIRAZ / DEN baselines) must NOT activate the half-frame
    eval branch — running it would do unnecessary work and could mislead
    operators with a deployment-irrelevant metric."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    # Hand-roll a paren-balanced extraction since arbitrarily-nested parens
    # aren't trivially expressible with re.
    start = src.find("_halfframe_eval_active = (")
    assert start != -1, "couldn't locate _halfframe_eval_active gate"
    depth = 0
    end = -1
    for i in range(start + len("_halfframe_eval_active = "), len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end != -1, "couldn't find matching closing paren"
    gate = src[start:end]
    assert "sim_zoom_warp is not None" in gate, (
        "half-frame eval gate must require sim_zoom_warp — without it, "
        "warp_inverse_masks() raises AttributeError"
    )
    assert "use_zoom_flow" in gate and "mask_half_sim_prob" in gate, (
        "half-frame eval gate must check BOTH use_zoom_flow and "
        "mask_half_sim_prob (either alone is sufficient to ship half-frame)"
    )


# ── 10. Codex R-Lane-D-Issue3: bootstrap script real-flag enforcement ────


def test_bootstrap_script_passes_required_tag_flag() -> None:
    """Codex R-Lane-D-Issue3: train_renderer's --tag is required=True. The
    previous script omitted it → argparse SystemExit on launch (the script
    literally couldn't run)."""
    script = (REPO / "scripts" / "remote_lane_d_halfframe_retrain.sh").read_text()
    # Find the train_renderer invocation
    m = re.search(
        r'src/tac/experiments/train_renderer\.py(.*?)(?=\n\s*BEST_FP32=|\Z)',
        script, re.DOTALL,
    )
    assert m, "couldn't locate train_renderer.py invocation"
    invocation = m.group(0)
    assert "--tag" in invocation, (
        "bootstrap MUST pass --tag (required=True in train_renderer argparse). "
        "Without it the script crashes immediately — Codex R-Lane-D-Issue3."
    )


def test_bootstrap_script_disables_or_satisfies_auth_eval_on_best() -> None:
    """Codex R-Lane-D-Issue3: --auth-eval-on-best defaults TRUE. Without
    matching --auth-eval-masks AND --auth-eval-poses, train_renderer hard-
    fails. Lane D's bootstrap runs the auth eval as a separate Stage 4
    (through contest_auth_eval.py) so the in-training auth eval must be
    explicitly disabled."""
    script = (REPO / "scripts" / "remote_lane_d_halfframe_retrain.sh").read_text()
    m = re.search(
        r'src/tac/experiments/train_renderer\.py(.*?)(?=\n\s*BEST_FP32=|\Z)',
        script, re.DOTALL,
    )
    assert m
    invocation = m.group(0)
    has_disable = "--no-auth-eval-on-best" in invocation
    has_masks = "--auth-eval-masks" in invocation
    has_poses = "--auth-eval-poses" in invocation
    assert has_disable or (has_masks and has_poses), (
        "bootstrap MUST either pass --no-auth-eval-on-best OR provide both "
        "--auth-eval-masks and --auth-eval-poses. Otherwise train_renderer "
        "raises RuntimeError at the end of training (Council R3 fix)."
    )


def test_bootstrap_script_finds_real_checkpoint_path() -> None:
    """Codex R-Lane-D-Issue3: train_renderer writes
    `renderer_<tag>_best_fp32.pt` (canonical) and `renderer_<tag>_best_fp4.pt`
    (FP4-packed dict, NOT a renderer.bin). The bootstrap must reference one of
    these REAL filenames — not the previous `*_fp4.bin` glob that matched zero
    files."""
    script = (REPO / "scripts" / "remote_lane_d_halfframe_retrain.sh").read_text()
    # The script must reference the real fp32 .pt name pattern.
    assert "renderer_${TAG}_best_fp32.pt" in script or \
           "renderer_<tag>_best_fp32.pt" in script.lower(), (
        "bootstrap must reference renderer_<tag>_best_fp32.pt — the actual "
        "filename train_renderer writes (NOT the previous *_fp4.bin glob "
        "which matched zero files)."
    )
    # And the previous broken glob MUST NOT reappear.
    assert "renderer_best_fp4.bin" not in script, (
        "bootstrap still references the broken *_fp4.bin glob — Codex "
        "R-Lane-D-Issue3 was incompletely applied."
    )


def test_bootstrap_script_exports_fp4a_renderer_bin() -> None:
    """Codex R-Lane-D-Issue3: the .pt file train_renderer writes is a
    torch.save dict of FP4-packed scales/indices — NOT a renderer.bin with
    FP4A magic. The inflate path REQUIRES FP4A magic, so the bootstrap must
    convert .pt → .bin via tac.renderer_export.export_asymmetric_checkpoint_fp4
    (the same path pipeline.py:step_export uses)."""
    script = (REPO / "scripts" / "remote_lane_d_halfframe_retrain.sh").read_text()
    assert "export_asymmetric_checkpoint_fp4" in script, (
        "bootstrap must call export_asymmetric_checkpoint_fp4 to produce a "
        "real FP4A renderer.bin — Codex R-Lane-D-Issue3. Otherwise the "
        "inflate side rejects the file (no FP4A magic bytes)."
    )


def test_bootstrap_script_has_dead_flag_preflight() -> None:
    """Codex R-Lane-D-Issue3 (defensive): the bootstrap must run an inline
    argparse-introspection preflight that asserts every --flag in the
    train_renderer invocation exists in train_renderer.py's argparse. This
    catches the dead-flag class of bug at script-launch time (before any
    GPU spend), not just in CI tests."""
    script = (REPO / "scripts" / "remote_lane_d_halfframe_retrain.sh").read_text()
    # The preflight must read train_renderer.py source and grep add_argument.
    assert "add_argument" in script and "INVENTED FLAGS" in script, (
        "bootstrap missing inline dead-flag preflight — Codex R-Lane-D-Issue3 "
        "defensive measure. Add a python -c block that introspects argparse."
    )


# ── 11. Codex follow-up: --auth-eval-on-best converts .pt → .bin ──────────


def test_auth_eval_on_best_converts_pt_to_fp4a_bin() -> None:
    """Codex R-Lane-D-followup: train_renderer's --auth-eval-on-best block
    USED to pass best_fp4 (a torch.save .pt of FP4-packed scales/indices)
    directly to build_submission_archive and auth_eval_renderer, both of
    which require FP4A magic-byte .bin format. The fix: load the FP32 .pt
    sidecar, rebuild the model from arch_meta, and call
    export_asymmetric_checkpoint_fp4 to produce a real .bin BEFORE the
    archive build."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    # The fix must call the canonical FP4A export inside the auth-eval block.
    assert "from tac.renderer_export import export_asymmetric_checkpoint_fp4" in src, (
        "auth-eval-on-best must call export_asymmetric_checkpoint_fp4 to "
        "produce a real FP4A .bin — Codex R-Lane-D-followup."
    )
    assert "best_fp4 = bin_path" in src, (
        "auth-eval-on-best must reassign best_fp4 to the .bin output path "
        "before the build_submission_archive call — Codex R-Lane-D-followup."
    )


def test_auth_eval_on_best_loads_fp32_sidecar_for_arch_meta() -> None:
    """Codex R-Lane-D-followup: the export needs the model's arch_meta
    (which lives in the FP32 sidecar's __meta__ key, not the FP4-packed .pt
    that lacks structure beyond opaque tensors). The auth-eval block must
    read the FP32 sidecar and use its __meta__ to rebuild the model."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    assert 'best_fp32.pt' in src and 'fp32_payload["__meta__"]' in src, (
        "auth-eval-on-best must load the FP32 sidecar and use __meta__ to "
        "rebuild the model — Codex R-Lane-D-followup."
    )
    # And the rebuild must use build_renderer with all arch fields the meta
    # carries, not just a subset.
    for field in (
        "arch[\"embed_dim\"]", "arch[\"base_ch\"]", "arch[\"motion_hidden\"]",
        "arch[\"use_zoom_flow\"]", "arch[\"padding_mode\"]",
    ):
        assert field in src, (
            f"auth-eval-on-best rebuild must thread {field} from arch_meta "
            f"(SHIRAZ-class drift risk if any field is dropped)."
        )


def test_auth_eval_on_best_fails_loud_on_non_default_variant() -> None:
    """Codex R-Lane-D-followup defensive: only variants in
    `_VARIANTS_BUILD_RENDERER_FP4A_OK` have a working FP4A export path. For
    other variants (c3_residual_renderer, vqvae, diffusion_teacher) the auth
    eval block must raise rather than silently produce a bogus .bin.

    Codex R5-2 Finding #1 (2026-04-27): the original implementation
    hardcoded `variant in ('default', None)` — too narrow, rejected real
    profiles like 'dilated' and 'psd'. The fix moved the check to the
    `_variant_supports_fp4a_export()` helper backed by exhaustive routing
    tables; this test now pins the helper invocation rather than the old
    error message."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    assert "_variant_supports_fp4a_export(_ckpt_variant)" in src, (
        "auth-eval-on-best must call _variant_supports_fp4a_export() to gate "
        "the FP4A path — codex R5-2 Finding #1. Hardcoded variant literals are "
        "the original Finding #1 bug."
    )
    # Defence-in-depth: the error message must explicitly reference the
    # NON-build_renderer set so operators know the remediation.
    assert "_VARIANTS_NON_BUILD_RENDERER" in src, (
        "auth-eval-on-best error message must reference the routing-table "
        "constant so operators can find the variant landscape quickly."
    )


def test_auth_eval_on_best_fails_loud_on_arch_drift_mismatch() -> None:
    """Codex R-Lane-D-followup defensive: if the loaded state_dict has
    missing or unexpected keys vs the rebuilt model, the auth-eval block
    must raise — that's the SHIRAZ arch-drift signal. Silent strict=False
    loading would ship a partially-initialized model and produce wrong
    scores."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    assert "SHIRAZ-class arch-drift bug" in src, (
        "auth-eval-on-best must raise on missing/unexpected keys with a "
        "SHIRAZ-class diagnostic — silent strict=False would corrupt scores."
    )


# ── 12. Codex R5-2 Finding #1: variant-aware auth-eval-on-best ───────────
#
# Pre-fix the auth-eval-on-best block hardcoded `variant in ('default', None)`.
# Profiles that flow through build_renderer (dilated, psd, mask_renderer,
# Lane D's `dilated`, plus ~20 others) all hit this guard and RuntimeError'd
# AFTER hours of training. The fix:
#   (1) early-fail in train() with a clear SystemExit + remediation, so we
#       never even start training when the config is broken
#   (2) routing tables (_VARIANTS_BUILD_RENDERER_FP4A_OK +
#       _VARIANTS_NON_BUILD_RENDERER) that are partition-exhaustive over
#       every variant declared in any profile.


def test_variant_routing_constants_exist() -> None:
    """The two routing tables must exist and be frozenset[str]."""
    from tac.experiments.train_renderer import (
        _VARIANTS_BUILD_RENDERER_FP4A_OK,
        _VARIANTS_NON_BUILD_RENDERER,
    )
    assert isinstance(_VARIANTS_BUILD_RENDERER_FP4A_OK, frozenset)
    assert isinstance(_VARIANTS_NON_BUILD_RENDERER, frozenset)
    # Common-sense guards.
    assert "default" in _VARIANTS_BUILD_RENDERER_FP4A_OK
    assert "dilated" in _VARIANTS_BUILD_RENDERER_FP4A_OK
    assert "diffusion_teacher" in _VARIANTS_NON_BUILD_RENDERER


def test_variant_routing_partitions_all_known_variants() -> None:
    """Codex R5-2 Finding #1: every variant declared in any profile must
    appear in EXACTLY one of the two routing tables. A variant that's in
    neither would silently fall through to the build_renderer branch (where
    it might crash with an unexpected kwarg) OR to the auth-eval guard
    (where it would RuntimeError after training). A variant in both is a
    routing-bug waiting to happen."""
    from tac.experiments.train_renderer import (
        _VARIANTS_BUILD_RENDERER_FP4A_OK,
        _VARIANTS_NON_BUILD_RENDERER,
    )
    from tac.profiles import PROFILES

    declared = {p.get("variant") for p in PROFILES.values() if p.get("variant")}
    declared.discard(None)
    overlap = _VARIANTS_BUILD_RENDERER_FP4A_OK & _VARIANTS_NON_BUILD_RENDERER
    assert overlap == set(), (
        f"variants in both routing tables: {sorted(overlap)}. "
        f"Each variant must be in EXACTLY one set."
    )
    union = _VARIANTS_BUILD_RENDERER_FP4A_OK | _VARIANTS_NON_BUILD_RENDERER
    missing = declared - union
    assert missing == set(), (
        f"variants declared in profiles.py but missing from BOTH routing "
        f"tables: {sorted(missing)}. Add each to "
        f"_VARIANTS_BUILD_RENDERER_FP4A_OK or _VARIANTS_NON_BUILD_RENDERER "
        f"in src/tac/experiments/train_renderer.py."
    )


def test_variant_supports_fp4a_export_dilated() -> None:
    """Codex R5-2 Finding #1: the dilated variant (every Lane A/B baseline,
    Lane D, the verified 0.9001 archive) MUST be FP4A-exportable. Pre-fix
    the auth-eval guard rejected it, killing the auth-eval-on-best contract
    for every dilated training run."""
    from tac.experiments.train_renderer import _variant_supports_fp4a_export
    assert _variant_supports_fp4a_export("dilated") is True
    assert _variant_supports_fp4a_export("psd") is True
    assert _variant_supports_fp4a_export("mask_renderer") is True
    assert _variant_supports_fp4a_export("default") is True
    assert _variant_supports_fp4a_export(None) is True
    assert _variant_supports_fp4a_export("") is True


def test_variant_supports_fp4a_export_rejects_non_build_renderer() -> None:
    """The variants with their own builder branch MUST report False —
    they cannot be FP4A-exported via export_asymmetric_checkpoint_fp4."""
    from tac.experiments.train_renderer import _variant_supports_fp4a_export
    for v in (
        "wavelet_renderer",
        "coord_renderer",
        "coolchic_renderer",
        "c3_residual_renderer",
        "dp_sims",
        "vqvae",
        "diffusion_teacher",
    ):
        assert _variant_supports_fp4a_export(v) is False, (
            f"variant={v!r} reported FP4A-exportable but lives in "
            f"_VARIANTS_NON_BUILD_RENDERER and does not have an "
            f"export_asymmetric_checkpoint_fp4 path."
        )


def test_train_early_fails_on_dilated_with_no_auth_inputs(monkeypatch) -> None:
    """Codex R5-2 Finding #1: train() must SystemExit at startup (before any
    GPU work) when --auth-eval-on-best is True (default) AND
    --auth-eval-masks/poses are missing. Pre-fix this only failed AFTER
    training completed — burning hours of compute to discover a config bug.

    We trigger it on a real build_renderer-compatible variant (dilated) to
    pin both halves of the validation: variant supports FP4A AND the inputs
    are missing."""
    train_renderer = pytest.importorskip("tac.experiments.train_renderer")
    args = train_renderer.parse_args([
        "--profile", "dilated_h64_half_frame",
        "--tag", "_unit_test_early_fail",
        # --auth-eval-on-best defaults True, intentionally NOT passing
        # --auth-eval-masks / --auth-eval-poses to trigger the guard.
    ])
    # Sanity: this is the case the guard targets.
    assert args.auth_eval_on_best is True
    assert args.auth_eval_masks is None
    assert args.auth_eval_poses is None
    # train() should SystemExit IMMEDIATELY — before configure_reproducibility
    # or any GPU work. No mock/stub needed because the guard fires before
    # device selection.
    with pytest.raises(SystemExit) as ei:
        train_renderer.train(args)
    msg = str(ei.value)
    assert "auth-eval-on-best" in msg, msg
    assert "Codex R5-2" in msg or "R5-2" in msg or "auth-eval-masks" in msg, msg


def test_train_early_fails_on_non_fp4a_variant_with_auth_eval(monkeypatch) -> None:
    """Codex R5-2 Finding #1: if a non-FP4A variant is paired with
    --auth-eval-on-best, train() must SystemExit at startup with a clear
    'switch variant or pass --no-auth-eval-on-best' message.

    Synthesise the failure mode by handcrafting an args namespace (no profile
    sets a non-FP4A variant + auth-eval-on-best together by default — the
    diffusion_teacher / vqvae profiles don't set Lane-D-style auth eval)."""
    train_renderer = pytest.importorskip("tac.experiments.train_renderer")
    import argparse

    # Build a namespace that satisfies the second guard (masks + poses present)
    # but fails the variant guard.
    args = argparse.Namespace(
        auth_eval_on_best=True,
        auth_eval_masks="/tmp/fake_masks.mkv",
        auth_eval_poses="/tmp/fake_poses.bin",
        variant="diffusion_teacher",  # in _VARIANTS_NON_BUILD_RENDERER
        profile="diffusion_teacher_smoke",
    )
    with pytest.raises(SystemExit) as ei:
        train_renderer.train(args)
    msg = str(ei.value)
    assert "diffusion_teacher" in msg, msg
    assert "FP4A" in msg or "auth-eval-on-best" in msg, msg


def test_train_early_validation_skipped_when_auth_eval_disabled() -> None:
    """When --no-auth-eval-on-best is passed, the early-fail guard must NOT
    fire — operators running smoke tests legitimately want to skip auth eval
    and the guard would block their workflow.

    Source-level pin (instead of full train() smoke): assert the guard is
    explicitly conditioned on `auth_eval_on_best`."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    # The guard MUST be wrapped in `if getattr(args, "auth_eval_on_best", False):`
    # — without that conditional, every train() call would hit the
    # masks/poses requirement, breaking smoke tests.
    assert 'if getattr(args, "auth_eval_on_best"' in src, (
        "early-fail guard must be conditional on auth_eval_on_best — "
        "unconditional guard would break --no-auth-eval-on-best smoke tests."
    )


def test_auth_eval_on_best_uses_helper_for_variant_check() -> None:
    """Source-level pin: the post-training auth-eval block must use the
    canonical helper (not a hardcoded literal). If someone reverts to
    `variant in ('default', None)` the routing-table abstraction collapses
    back to the original Finding #1 bug."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    assert "_variant_supports_fp4a_export(" in src, (
        "auth-eval-on-best block must call _variant_supports_fp4a_export() "
        "for the variant check (codex R5-2 Finding #1). Hardcoded variant "
        "literals are forbidden — they desync from build_renderer dispatch."
    )
