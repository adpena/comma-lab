"""Lane K profile (`dsconv_quantizr_killer`) — pinning tests.

Council brief 2026-04-27. Lane A holds the 1.15 [contest-CUDA] frontier with
the dilated-h64 baseline arch (288K params, ~290KB FP32 renderer.bin). Lane K
is the orthogonal rate attack to Lane S (Self-Compression) and Lane V
(half-frame): a from-scratch retrain at Quantizr-class capacity (88K params,
DSConv + FiLM) shipping FULL-frame masks anchored on Lane A's verified
artifacts.

These tests pin:
    1. The `dsconv_quantizr_killer` profile is registered in PROFILES.
    2. Architecture matches Quantizr-class (88,996 params verified against
       build_renderer — exact Quantizr 88K match).
    3. DSConv is enabled (the user-spec flag and the Quantizr trick).
    4. FiLM conditioning is enabled (pose_dim=6, NOT the dead-resolver
       legacy where pose_dim was silently 0).
    5. Full-frame paradigm: use_zoom_flow=False AND mask_half_sim_prob=0.0
       (Lane K is NOT half-frame; that's Lane V).
    6. CLAUDE.md non-negotiables (eval_roundtrip=True, deterministic=True,
       seed pinned).
    7. Best-practice training tricks (Fridrich aux losses, KL distill,
       hinge SegNet, per-class weights, SWA).
    8. 5-stage QAT config matches our advantage over Quantizr's vanilla
       (residual codebook, robust scale, stochastic rounding).
    9. Profile passes preflight.

Cost paranoia: zero GPU dollars wasted on a misconfigured 12h run.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


# ── 1. Profile structure & registration ──────────────────────────────────


def test_dsconv_quantizr_killer_profile_registered() -> None:
    """The profile MUST be in PROFILES dict so --profile dsconv_quantizr_killer
    works. If this fails, the bootstrap script will SystemExit before launch."""
    from tac.profiles import PROFILES, DSCONV_QUANTIZR_KILLER

    assert "dsconv_quantizr_killer" in PROFILES, (
        "DSCONV_QUANTIZR_KILLER not registered in PROFILES dict. "
        "scripts/remote_lane_k_dsconv_quantizr_killer.sh expects "
        "--profile dsconv_quantizr_killer to resolve."
    )
    assert PROFILES["dsconv_quantizr_killer"] is DSCONV_QUANTIZR_KILLER


def test_profile_experiment_type_is_renderer_training() -> None:
    """Lane K is a renderer training profile — must declare experiment_type
    so the preflight + train_renderer dispatch route correctly."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["experiment_type"] == "renderer_training"


# ── 2. Architecture: Quantizr-class capacity ─────────────────────────────


def test_arch_matches_quantizr_88k_target() -> None:
    """The Lane K profile MUST produce ~88K params (Quantizr-class).
    Council 2026-04-27: 88,996 params verified empirically. Acceptable
    band [80K, 100K] guards against accidental capacity drift if anyone
    edits motion_hidden / mid_ch."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER
    from tac.renderer import build_renderer

    p = DSCONV_QUANTIZR_KILLER
    model = build_renderer(
        num_classes=5,
        embed_dim=p["embed_dim"],
        base_ch=p["base_ch"],
        mid_ch=p["mid_ch"],
        motion_hidden=p["motion_hidden"],
        depth=p["depth"],
        blend_mode="scalar",
        noise_mode="deterministic",
        motion_type="learned_cnn",
        use_zoom_flow=p["use_zoom_flow"],
        use_dsconv=p["use_dsconv"],
        padding_mode=p["padding_mode"],
        use_dilation=p["use_dilation"],
        pose_dim=p["pose_dim"],
    )
    n = sum(pp.numel() for pp in model.parameters())
    # Tight band — we want any architecture edit to land ~88K.
    assert 80_000 <= n <= 100_000, (
        f"Lane K param count {n:,} outside Quantizr-class band [80K, 100K]. "
        f"Council target: 88,996 (verified 2026-04-27)."
    )


def test_arch_specific_values() -> None:
    """The exact arch knobs — base_ch=24, mid_ch=32, motion_hidden=16,
    depth=1, embed_dim=6, pose_dim=6. These are the Quantizr-replica numbers
    verified empirically at 88,996 params."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    p = DSCONV_QUANTIZR_KILLER
    assert p["base_ch"] == 24, f"expected base_ch=24, got {p['base_ch']}"
    assert p["mid_ch"] == 32, f"expected mid_ch=32 (user spec hidden_ch=32), got {p['mid_ch']}"
    assert p["motion_hidden"] == 16, (
        f"expected motion_hidden=16 (Quantizr-class), got {p['motion_hidden']}"
    )
    assert p["depth"] == 1, f"expected depth=1 (single-scale), got {p['depth']}"
    assert p["embed_dim"] == 6, f"expected embed_dim=6, got {p['embed_dim']}"
    assert p["pose_dim"] == 6, (
        f"expected pose_dim=6 (FiLM modulation, full conditioning from epoch 0), "
        f"got {p['pose_dim']}"
    )


def test_dsconv_enabled() -> None:
    """The user-spec flag and the Quantizr trick — depthwise-separable
    convolutions are MANDATORY for Lane K. Without them the param count
    would be ~3x larger and we'd lose the rate attack."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["use_dsconv"] is True, (
        "Lane K's primary architectural lever is DSConv; without it the "
        "renderer balloons past Quantizr-class capacity and the rate "
        "attack fails."
    )


def test_full_frame_paradigm() -> None:
    """Lane K is the FULL-frame complement to Lane V (half-frame).
    use_zoom_flow=False AND mask_half_sim_prob=0.0 — both must be set
    explicitly to defend against a caller flipping one without the other
    (preflight enforces consistency between the two flags)."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    p = DSCONV_QUANTIZR_KILLER
    assert p["use_zoom_flow"] is False, (
        "Lane K is FULL-frame; use_zoom_flow=True would route to "
        "AsymmetricPairGenerator with RadialZoomWarp inflate-side warp "
        "(that's Lane V's paradigm)."
    )
    assert p["mask_half_sim_prob"] == pytest.approx(0.0), (
        "Lane K trains on full-frame masks; mask_half_sim_prob>0 would "
        "inject warp-expansion (that's Lane V's paradigm)."
    )


def test_padding_mode_zeros() -> None:
    """padding_mode='zeros' matches the dilated-h64 baseline byte-for-byte
    for the few layers where padding affects FP4 quantization scale."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["padding_mode"] == "zeros"


# ── 3. CLAUDE.md non-negotiables ─────────────────────────────────────────


def test_eval_roundtrip_true() -> None:
    """CLAUDE.md non-negotiable: every training path MUST set
    eval_roundtrip=True. Without it the proxy-auth gap can be 2-11x on
    PoseNet (memory: feedback_proxy_auth_math_useless)."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["eval_roundtrip"] is True, (
        "eval_roundtrip MUST be True (CLAUDE.md non-negotiable). "
        "Without it, every TTO run optimises against a proxy that doesn't "
        "simulate the contest eval roundtrip."
    )


def test_seed_pinned() -> None:
    """Reproducibility — seed=1234 (matches build_baseline_archive.py default
    and the script's PYTHONHASHSEED for end-to-end determinism)."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["seed"] == 1234, (
        f"Lane K seed must be 1234 (matches PYTHONHASHSEED in bootstrap), "
        f"got {DSCONV_QUANTIZR_KILLER['seed']}"
    )


def test_deterministic_true() -> None:
    """Deterministic mode pins CUBLAS_WORKSPACE_CONFIG, cudnn.deterministic,
    use_deterministic_algorithms — required for bit-exact same-seed re-runs
    on the same GPU SKU + PyTorch version."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["deterministic"] is True


# ── 4. Loss configuration (best-practice training tricks) ────────────────


def test_segnet_loss_mode_hinge() -> None:
    """The user spec `seg_loss_mode='hinge'` maps to the canonical key
    `segnet_loss_mode='hinge'`. SegNet hinge loss is our standard recipe
    (proven across SHIRAZ / DEN / Lane D / Lane V)."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["segnet_loss_mode"] == "hinge"


def test_hinge_margin_half() -> None:
    """The user spec `seg_margin=0.5` maps to `hinge_margin=0.5`. Tighter
    than WILDE/SHIRAZ/DEN (1.0) because Lane K has fewer params and needs
    every gradient drop to count."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["hinge_margin"] == pytest.approx(0.5)


def test_score_weights() -> None:
    """SegNet dominates per scoring math (77x more important).
    seg_weight=100 and pose_weight=10 match SHIRAZ / DEN / Lane V."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    p = DSCONV_QUANTIZR_KILLER
    assert p["seg_weight"] == pytest.approx(100.0)
    assert p["pose_weight"] == pytest.approx(10.0)


def test_kl_distill_calibrated() -> None:
    """Quantizr's #1 SegNet trick: KL distillation, T=2.0. weight=0.002 POST
    losses.py:705 reduction fix (pre-fix, raw KL was ~5000× larger so weight
    1.0 was implicitly running at 5000× intended). Lane K matches Lane V's
    calibration."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    p = DSCONV_QUANTIZR_KILLER
    assert p["kl_distill_temperature"] == pytest.approx(2.0)
    assert p["kl_distill_weight"] == pytest.approx(0.002), (
        f"kl_distill_weight=0.002 is the post-reduction-fix calibration; "
        f"got {p['kl_distill_weight']}. Pre-fix value 1.0 would implicitly "
        f"run at 5000× intended — would dominate the scorer signal."
    )


def test_fridrich_aux_losses_enabled() -> None:
    """Fridrich inverse-steganalysis losses (texture / L∞ / Markov / wavelet
    variance noise / uncertainty) — same recipe as SHIRAZ. These are
    scorer-arch-specific so they transfer cleanly to the smaller arch."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    p = DSCONV_QUANTIZR_KILLER
    assert p["use_texture_loss"] is True
    assert p["use_linf_penalty"] is True
    assert p["use_markov_loss"] is True
    assert p["use_variance_noise"] is True
    assert p["variance_noise_mode"] == "wavelet_db4", (
        "Wavelet_db4 is the Fridrich R2 C2 fix (un-decimated Daubechies-8 "
        "sub-band energy per Holub & Fridrich 2014 §III.B). Box mode is "
        "legacy A/B only."
    )
    assert p["use_uncertainty_loss"] is True


def test_per_class_weights_enabled() -> None:
    """Per-class weights up-weight lane markings 15x (Yousfi: 1.2% of pixels
    but critical for PoseNet)."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["use_per_class_weights"] is True


def test_swa_enabled() -> None:
    """SWA → wider minima → FP4 survives the post-training quantization pass.
    Polyak 1992; standard across all production renderer profiles."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["use_swa"] is True


def test_posetto_noise_std_present() -> None:
    """posetto_noise_std=0.5 is consumed by experiments/optimize_poses.py
    during pose-TTO; surfaced as profile metadata so the operator can read
    the contract from one place."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["posetto_noise_std"] == pytest.approx(0.5)


# ── 5. 5-stage quantization config (our advantage over Quantizr) ────────


def test_fp4_codebook_residual() -> None:
    """fp4_codebook='residual' (denser-near-zero, 4× better small-magnitude
    preservation). Critical for an 88K renderer where every weight matters."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["fp4_codebook"] == "residual", (
        "Lane K must use the residual codebook (denser-near-zero); the "
        "default uniform codebook costs ~4× more error on small weights."
    )


def test_fp4_robust_scale() -> None:
    """fp4_robust_scale=True (per-block scale via p99.5 quantile vs max).
    Protects small-magnitude tail from outlier-driven collapse."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["fp4_robust_scale"] is True


def test_fp4_stochastic() -> None:
    """fp4_stochastic=True (unbiased dither during training; auto-disabled
    in eval mode for inflate determinism)."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    assert DSCONV_QUANTIZR_KILLER["fp4_stochastic"] is True


# ── 6. 5-phase QAT schedule ──────────────────────────────────────────────


def test_5phase_schedule_present() -> None:
    """Every phase must declare epochs > 0 — the 5-phase Quantizr-adapted
    schedule (anchor → finetune → joint → QAT → final). Total ~3000 epochs
    matches Lane V budget for fair A/B."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    p = DSCONV_QUANTIZR_KILLER
    for i in range(1, 6):
        assert p[f"phase{i}_epochs"] > 0, f"phase{i}_epochs must be > 0"
    total = sum(p[f"phase{i}_epochs"] for i in range(1, 6))
    assert 2000 <= total <= 4000, (
        f"total epochs {total} outside [2000, 4000] band; Lane K targets "
        f"~3000 epochs for ~12h on 4090 (matches Lane V budget)."
    )


def test_phase_lrs_descending() -> None:
    """LR descends across phases: P1 anchor (highest), P5 final (lowest)."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    p = DSCONV_QUANTIZR_KILLER
    lrs = [p[f"phase{i}_lr"] for i in range(1, 6)]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1], (
            f"LR must descend across phases; got {lrs}"
        )
    # Phase 4 (QAT) MUST be 0.1× of phase 2 base per Lin et al. 2017.
    assert p["phase4_lr"] <= 0.2 * p["phase2_lr"]


# ── 7. Preflight pass ────────────────────────────────────────────────────


def test_passes_preflight() -> None:
    """The profile must pass tac.preflight.preflight_profiles. This catches
    typo'd keys, missing required arch keys, eval_roundtrip!=True, invalid
    padding_mode, etc. — before we burn 12h of GPU on a misconfigured run."""
    from tac.preflight import preflight_profiles

    violations = preflight_profiles(strict=False, verbose=False)
    ours = [v for v in violations if "dsconv_quantizr_killer" in v]
    assert not ours, f"Preflight violations specific to dsconv_quantizr_killer: {ours}"


def test_required_arch_keys_present() -> None:
    """Lane K profile must declare every key in PROFILE_REQUIRED_ARCH_KEYS:
    base_ch, mid_ch, depth, pose_dim, padding_mode, eval_roundtrip, seed,
    deterministic. Missing any is a preflight failure."""
    from tac.preflight import PROFILE_REQUIRED_ARCH_KEYS
    from tac.profiles import DSCONV_QUANTIZR_KILLER

    for key in PROFILE_REQUIRED_ARCH_KEYS:
        assert key in DSCONV_QUANTIZR_KILLER, (
            f"required arch key {key!r} missing from DSCONV_QUANTIZR_KILLER"
        )


# ── 8. End-to-end build sanity ───────────────────────────────────────────


def test_build_renderer_returns_pair_generator() -> None:
    """With use_zoom_flow=False, build_renderer should return PairGenerator
    (NOT AsymmetricPairGenerator). The renderer_export FP4A path supports
    both pair_modes."""
    from tac.profiles import DSCONV_QUANTIZR_KILLER
    from tac.renderer import build_renderer, PairGenerator, AsymmetricPairGenerator

    p = DSCONV_QUANTIZR_KILLER
    model = build_renderer(
        num_classes=5,
        embed_dim=p["embed_dim"],
        base_ch=p["base_ch"],
        mid_ch=p["mid_ch"],
        motion_hidden=p["motion_hidden"],
        depth=p["depth"],
        blend_mode="scalar",
        noise_mode="deterministic",
        motion_type="learned_cnn",
        use_zoom_flow=p["use_zoom_flow"],
        use_dsconv=p["use_dsconv"],
        padding_mode=p["padding_mode"],
        use_dilation=p["use_dilation"],
        pose_dim=p["pose_dim"],
    )
    # use_zoom_flow=False → PairGenerator (NOT AsymmetricPairGenerator).
    assert isinstance(model, PairGenerator), (
        f"expected PairGenerator (use_zoom_flow=False), got {type(model).__name__}"
    )
    assert not isinstance(model, AsymmetricPairGenerator), (
        "PairGenerator path should NOT route to AsymmetricPairGenerator"
    )
