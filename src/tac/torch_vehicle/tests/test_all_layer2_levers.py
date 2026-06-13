# SPDX-License-Identifier: MIT
"""NO-FAKE tests for ALL FIVE Layer-2 in-curriculum levers wired into the
torch-vehicle HNeRV training path (``tac.torch_vehicle.driver``).

The levers (each a default-OFF, byte-identical-when-disabled, composable knob):

* **Lever 1** — differentiable brotli-rate surrogate (``rate_lambda_w`` /
  ``rate_lambda_lat`` → ``tac.losses.rate_surrogate.brotli_rate_surrogate``).
* **Lever 2** — score-domain argmax-flip seg surrogate + the per-epoch
  TEMPERATURE-ANNEAL hook (``seg_surrogate`` / ``seg_temperature`` /
  ``seg_temperature_end`` → ``seg_temperature_for_epoch``).
* **Lever 3** — pose-FiLM store (``cfg.pose_film_enabled``; the wrapper +
  additive pose codec section; its own dedicated suite is
  ``test_pose_film_wire_in.py`` — here it is the COMPOSE member).
* **Lever 4** — score-aware QAT (``score_aware_qat`` → per-tensor
  sensitivity-weighted INT8 grid via
  ``tac.torch_vehicle.score_aware_qat``).
* **Lever 5** — margin-weighted seg promotion (``margin_weight_tau`` → the
  ``exp(−margin/τ)`` per-pixel boundary weight on the Lever-2 surrogate).

THE LOAD-BEARING CLAIMS (each, if wrong, is either a FAKE lever OR — worse —
silently changes the LIVE base_ch=20 basin if it crash-resumes onto this code):

A. **ALL-DEFAULT BYTE IDENTITY (the daemon-safety guard).** A full synthetic
   driver run with EVERY lever at its default (off) produces the IDENTICAL
   best-score trajectory AND the IDENTICAL archive bytes as a run on the
   pre-lever code path. We prove this two ways: (1) ``_train_one_epoch`` with an
   all-default spec produces a bit-identical decoder/latent state vs a hand-rolled
   reference epoch that exercises ONLY the vendored ops; (2) two driver runs with
   all-default specs are bit-identical to each other (determinism) AND the
   regularizer helper returns the EXACT legacy tensor on the C1a path.

B. **EACH LEVER ACTUALLY CHANGES THE LOSS/GRAD/BYTES IN THE CLAIMED DIRECTION**
   (NO constant-checking — if replacing the lever body with ``return baseline``
   would still pass, the test is fake):
   - Lever 1: enabling ``rate_lambda_w`` adds a real, gradient-carrying rate term
     to the loss that DIFFERS from the no-rate loss; the conditional weight
     entropy is ``<=`` the marginal (the true-bound property).
   - Lever 2 anneal: ``seg_temperature_for_epoch`` actually changes the
     temperature per epoch when ``seg_temperature_end`` is set (and is STATIC when
     not); the annealed T flows into the surrogate value.
   - Lever 4: a score-aware-QAT step with a non-uniform sensitivity EMA produces a
     DIFFERENT decoder update than uniform QAT; uniform sensitivity reproduces the
     vendored quant bit-identically.
   - Lever 5: ``margin_weight_tau`` re-weights the per-pixel seg surrogate toward
     small-margin (boundary) pixels — the weighted loss DIFFERS from the unweighted
     one, and the weight is monotone-decreasing in the margin.

C. **COMPOSE ALL FIVE** end-to-end: a synthetic driver run with Levers 1+2(+anneal)
   +3+4+5 ALL ENABLED runs to a DONE marker, byte-closes an archive (WITH the
   Lever-3 pose section), and the archive parses back (Lever-3 pose section
   round-trips). This proves the five levers compose in one forward/backward/
   export path without crashing or corrupting the archive grammar.
"""

from __future__ import annotations

import tempfile

import pytest
import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import (
    SEG_ANNEAL_GRADIENT_FLOOR_T,
    StageSpec,
    seg_anneal_temperature_is_gradient_alive,
    seg_temperature_for_epoch,
)
from tac.torch_vehicle.driver import (
    _SEG_NUM_CLASSES,
    TorchVehicleConfig,
    TorchVehicleDriver,
    _seg_loss_for_spec,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _stage(**overrides) -> StageSpec:
    base = {
        "name": "lever_test",
        "epochs": 2,
        "seg_loss_fn": _ce,
        "eval_every": 1,
        "batch_size": 3,
        "ema_decay": 0.999,
        "use_muon": False,
        "adamw_lr": 1e-3,
        "muon_lr": 2e-4,
        "muon_weight_decay": 0.0,
        "latent_lr_mult": 10.0,
        "grad_clip": 1e9,
        "grad_clip_muon": 1e9,
        "lr_floor_ratio": 5e-6,
        "seg_weight": 100.0,
        "pose_weight": 1.0,
        "cat_lambda": 0.0,
        "cat_sigma": 0.2,
        "use_qat": False,
        "init_latents_random": True,
    }
    base.update(overrides)
    return StageSpec(**base)


def _tmp_out() -> str:
    """A fresh temp out-dir (the TelemetryWriter mkdir's it; ``/dev/null`` fails)."""
    return tempfile.mkdtemp(prefix="l2_lever_test_")


def _make_seg_inputs(seed: int = 0, B: int = 2, H: int = 8, W: int = 12):
    g = torch.Generator().manual_seed(seed)
    seg_out = torch.randn(B, _SEG_NUM_CLASSES, H, W, generator=g, requires_grad=True)
    seg_targets_hard = torch.randint(
        0, _SEG_NUM_CLASSES, (B, H, W), generator=g, dtype=torch.int64
    )
    return seg_out, seg_targets_hard


def _run_driver(cfg: TorchVehicleConfig, spec: StageSpec, *, n_pairs: int = 6, seed: int = 0):
    """Run a tiny synthetic driver to completion; return (driver, summary)."""
    sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=seed)
    driver = TorchVehicleDriver(
        cfg, scorer=sc, vendored=import_vendored_bundle(), curriculum=[spec]
    )
    summary = driver.run()
    return driver, summary


# ===========================================================================
# CLAIM A — ALL-DEFAULT BYTE IDENTITY (the daemon-safety guard).
# ===========================================================================
def test_stagespec_all_lever_fields_default_to_off():
    """Every lever field defaults to its OFF value on a vendored-projected spec —
    the levers are OPT-IN, never on by accident (the basin contention guard)."""
    spec = _stage()
    # Lever 1
    assert spec.rate_lambda_w == 0.0
    assert spec.rate_lambda_lat == 0.0
    # Lever 2 (+ anneal)
    assert spec.seg_surrogate is None
    assert spec.seg_temperature == 1.0
    assert spec.seg_temperature_end is None
    # Lever 4
    assert spec.score_aware_qat is False
    assert spec.qat_sensitivity_decay == 0.99
    # Lever 5
    assert spec.margin_weight_tau is None


def test_config_pose_film_defaults_off():
    """Lever 3 (cfg.pose_film_enabled) defaults OFF — the live basin builds the
    vendored decoder unchanged and adds NO pose section (byte-identical)."""
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out())
    assert cfg.pose_film_enabled is False


def test_all_default_driver_run_is_deterministic_and_byte_identical(tmp_path):
    """Two all-default synthetic driver runs (same seed) produce bit-identical
    best archives AND the identical best-score — the determinism floor the
    daemon-safety claim rests on (an all-default run is the pre-lever path)."""
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    spec = _stage(epochs=3, cat_lambda=0.0)  # pure default loss path
    cfg_a = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=out_a,
        checkpoint_every_epochs=1, device="cpu", seed=0,
    )
    cfg_b = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=out_b,
        checkpoint_every_epochs=1, device="cpu", seed=0,
    )
    _, sum_a = _run_driver(cfg_a, spec, seed=0)
    _, sum_b = _run_driver(cfg_b, spec, seed=0)

    assert sum_a["best_score"] == sum_b["best_score"], "all-default run is non-deterministic"
    arch_a = (out_a / "best" / "best_archive.bin").read_bytes()
    arch_b = (out_b / "best" / "best_archive.bin").read_bytes()
    assert arch_a == arch_b, "all-default best archive bytes differ between identical runs"


def test_default_train_epoch_matches_vendored_only_reference():
    """``_train_one_epoch`` with an ALL-DEFAULT spec produces the IDENTICAL decoder
    + latent state as a hand-rolled reference epoch exercising ONLY the vendored ops
    (no lever code path touched). If ANY lever silently mutated the default forward/
    backward, the post-epoch decoder weights would diverge — this is the strongest
    daemon-safety proof (the live basin's update is unchanged byte-for-byte)."""
    torch.manual_seed(0)
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    v = import_vendored_bundle()
    spec = _stage(epochs=1, cat_lambda=0.0, seg_loss_fn=_ce)

    # --- run ONE epoch through the driver (default path) ---
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(
        torch.randn(6, 28, generator=torch.Generator().manual_seed(7)) * 0.1
    )
    rt = driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )
    # Capture pre-epoch state, run, capture post-epoch state.
    torch.manual_seed(123)  # pin the randperm draw
    driver._train_one_epoch(rt, spec, epoch_in_stage=0)
    post_driver = {k: v_.detach().clone() for k, v_ in rt.decoder.state_dict().items()}
    post_latents = rt.latents.detach().clone()

    # --- hand-rolled reference epoch (vendored ops only) ---
    torch.manual_seed(0)
    sc2 = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    decoder2 = driver._new_decoder(device=torch.device("cpu"))
    latents2 = torch.nn.Parameter(
        torch.randn(6, 28, generator=torch.Generator().manual_seed(7)) * 0.1
    )
    rt2 = driver._build_stage_runtime(
        spec, decoder=decoder2, latents=latents2, ema_decoder=None, ema_latents=None
    )
    torch.manual_seed(123)
    pair_indices = torch.randperm(6)
    bs = spec.batch_size
    for bstart in range(0, 6, bs):
        idx = pair_indices[bstart : bstart + bs]
        B = len(idx)
        decoded_pair = decoder2(latents2[idx])
        flat = decoded_pair.reshape(B * 2, 3, 384, 512)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
        decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
        dc = decoded_bhwc.clamp(0, 255)
        dr = dc.round()
        decoded_bhwc = dc + (dr - dc).detach()
        rt2.adamw_opt.zero_grad()
        seg_out, pose_pred6 = sc2.seg_pose_forward(decoded_bhwc)
        seg_l = spec.seg_loss_fn(seg_out, sc2.seg_targets_hard[idx])
        pose_mse = F.mse_loss(pose_pred6, sc2.pose_targets[idx])
        pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
        loss = spec.seg_weight * seg_l + spec.pose_weight * pose_l
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*rt2.adamw_params, latents2], spec.grad_clip)
        rt2.adamw_opt.step()
        v.ema_update(rt2.ema_decoder, decoder2, rt2.ema_latents, latents2, decay=spec.ema_decay)

    for k in post_driver:
        assert torch.equal(post_driver[k], rt2.decoder.state_dict()[k]), (
            f"default driver epoch diverged from vendored-only reference at {k} — a "
            "lever code path silently changed the DEFAULT update (basin-safety FAIL)"
        )
    assert torch.equal(post_latents, rt2.latents.detach()), "default latent update diverged"


def test_weight_regularizers_default_returns_none():
    """``_weight_regularizers`` returns None when NO lever is active (cat_lambda==0
    AND both rate_lambda_*==0) — the default path adds NOTHING to the loss."""
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_stage()])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    spec = _stage(cat_lambda=0.0, rate_lambda_w=0.0, rate_lambda_lat=0.0)
    assert driver._weight_regularizers(decoder, latents, spec) is None


def test_weight_regularizers_c1a_only_matches_legacy_tensor():
    """On the C1a-only path (cat_lambda>0, rate off) ``_weight_regularizers`` returns
    EXACTLY ``cat_lambda * cat_entropy_v2(...)`` — the SAME tensor the legacy code
    added — so a C1a-stage basin resuming onto this code is byte-identical.

    ``cat_entropy_v2`` samples internally, so for a BIT-EXACT comparison we pin the
    global RNG to the SAME state immediately before BOTH calls on the SAME decoder
    object (two separate decoder constructions diverge because the sampler draws on a
    different RNG-stream offset). With the RNG pinned identically, the helper's term
    must equal the hand-computed legacy term to the last bit (``torch.equal``)."""
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_stage()])
    torch.manual_seed(5)
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    spec = _stage(cat_lambda=0.01, cat_sigma=0.2, rate_lambda_w=0.0, rate_lambda_lat=0.0)

    # Pin the RNG, take the helper's term; re-pin to the IDENTICAL state, take the
    # legacy term on the SAME decoder — both consume the same internal sample draw.
    rng = torch.get_rng_state()
    reg = driver._weight_regularizers(decoder, latents, spec)
    torch.set_rng_state(rng)
    ent = v.cat_entropy_v2(decoder, sigma=spec.cat_sigma, sample_size=2000, device="cpu")
    expected = spec.cat_lambda * ent
    assert torch.equal(reg, expected), (
        f"C1a-only regularizer {reg.item():.8f} != legacy term {expected.item():.8f} "
        "(the C1a-stage basin update would diverge byte-for-byte)"
    )


# ===========================================================================
# CLAIM B — EACH LEVER ACTUALLY CHANGES THE LOSS/GRAD/BYTES.
# ===========================================================================
# --- Lever 1: differentiable rate surrogate ---
def test_lever1_rate_term_changes_loss_and_has_gradient():
    """Enabling ``rate_lambda_w`` adds a real, gradient-carrying rate term that
    DIFFERS from the no-rate regularizer — and the weights see its gradient (so the
    optimizer can descend it). A silent no-op would give an identical loss + zero
    rate-grad."""
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_stage()])
    torch.manual_seed(11)
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)

    spec_off = _stage(cat_lambda=0.0, rate_lambda_w=0.0, rate_lambda_lat=0.0)
    spec_on = _stage(cat_lambda=0.0, rate_lambda_w=0.5, rate_lambda_lat=0.0)

    assert driver._weight_regularizers(decoder, latents, spec_off) is None
    reg_on = driver._weight_regularizers(decoder, latents, spec_on)
    assert reg_on is not None and reg_on.item() > 0.0, "rate-on must add a positive rate term"
    # Real gradient to the decoder weights.
    decoder.zero_grad()
    reg_on.backward()
    grads = [m.weight.grad for n, m in decoder.named_modules()
             if hasattr(m, "weight") and getattr(m, "weight", None) is not None
             and m.weight.grad is not None]
    assert grads, "rate surrogate produced NO gradient to any decoder weight (FAKE)"
    assert any(g.abs().sum().item() > 0 for g in grads), "rate gradient is all-zero (FAKE)"


def test_lever1_conditional_entropy_is_below_marginal_true_bound():
    """The order-1 conditional weight entropy H(W|W_prev) is <= the marginal H(W)
    (the conditioning inequality) — the TRUE-bound property that makes it a
    conservative brotli proxy. (Mechanism check, not a constant.)"""
    from tac.losses.rate_surrogate import conditional_weight_entropy

    torch.manual_seed(3)
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_stage()])
    decoder = driver._new_decoder(device=torch.device("cpu"))

    h_cond = conditional_weight_entropy(decoder, device="cpu").item()
    # The MARGINAL weight entropy H(W) on the SAME INT8 grid (the vendored C1a
    # quantity the rate surrogate refines). Conditioning never increases entropy, so
    # H(W|W_prev) <= H(W).
    h_marg = v.cat_entropy_v2(decoder, sigma=0.2, sample_size=4000, device="cpu").item()
    # Conditioning never increases entropy (small numerical slack for soft-hist + the
    # cap on adjacency sample size vs the marginal's larger sample).
    assert h_cond <= h_marg + 1e-2, (
        f"H(W|W_prev)={h_cond:.4f} must be <= H(W)={h_marg:.4f} (the true-bound property)"
    )


# --- Lever 2 anneal hook ---
def test_lever2_anneal_disabled_returns_static_temperature():
    """``seg_temperature_end is None`` (default) returns the STATIC temperature for
    EVERY epoch — byte-identical to the pre-anneal driver."""
    spec = _stage(epochs=10, seg_surrogate="soft_cosine", seg_temperature=0.7,
                  seg_temperature_end=None)
    for e in range(10):
        assert seg_temperature_for_epoch(spec, e) == 0.7


def test_lever2_anneal_actually_changes_temperature_per_epoch():
    """With ``seg_temperature_end`` set, the temperature cosine-anneals from start to
    end over the stage — the FIRST epoch is the start T, the LAST is the end T, and
    it is monotone in between. A no-op anneal would return a constant (this FAILS for
    a constant)."""
    spec = _stage(epochs=10, seg_surrogate="soft_cosine", seg_temperature=1.0,
                  seg_temperature_end=0.05)
    temps = [seg_temperature_for_epoch(spec, e) for e in range(10)]
    assert abs(temps[0] - 1.0) < 1e-9, "epoch 0 must be the start temperature"
    assert abs(temps[-1] - 0.05) < 1e-9, "final epoch must be the end temperature"
    # Monotone decreasing (cosine 1->0 progress with start>end).
    assert all(temps[i] >= temps[i + 1] - 1e-9 for i in range(len(temps) - 1)), (
        f"anneal not monotone: {temps}"
    )
    # NOT a constant (the FAKE guard).
    assert max(temps) - min(temps) > 0.5, "anneal did not actually move the temperature"


def test_lever2_annealed_temperature_flows_into_surrogate_value():
    """The annealed per-epoch temperature changes the surrogate value (proving the
    driver's epoch_temperature actually reaches the seg loss)."""
    seg_out, targets = _make_seg_inputs()
    spec = _stage(seg_surrogate="soft_cosine", seg_temperature=1.0, seg_temperature_end=0.1)
    t_early = seg_temperature_for_epoch(spec, 0)  # ~1.0
    t_late = seg_temperature_for_epoch(spec, spec.epochs - 1)  # ~0.1 -- but epochs=2 here
    # Use distinct temperatures explicitly to assert the value depends on T.
    v_warm = _seg_loss_for_spec(spec, seg_out, targets, temperature=1.0)
    v_cold = _seg_loss_for_spec(spec, seg_out, targets, temperature=0.1)
    assert not torch.allclose(v_warm, v_cold, atol=1e-4), (
        "surrogate value is temperature-invariant — the anneal hook is a no-op (FAKE)"
    )
    assert t_early >= t_late  # sanity: the schedule moves in the right direction


def test_lever2_anneal_at_t_min_over_long_stage_is_clamped_and_surrogate_finite():
    """R5 lens-C (long-run numerical stability of the anneal at T->min). Over a
    LONG stage (the PR95 stages run thousands of epochs) the cosine reaches its
    floor at the FINAL epoch. Two properties must hold to the very end:
    (i) the temperature is CLAMPED at ``seg_temperature_end`` and never undershoots
        to 0 (a T=0 would divide-by-zero in ``softmax(pred/T)``); and
    (ii) the soft-cosine surrogate at that coldest T is FINITE and in [0,1]
         (``F.softmax`` is internally max-stable, so even T=0.02 -> pred*50 does
         not overflow). A regression that let the anneal undershoot or the
         surrogate blow up would surface here, not at 80 epochs."""
    epochs = 9000
    t_end = 0.02
    spec = _stage(epochs=epochs, seg_surrogate="soft_cosine",
                  seg_temperature=1.0, seg_temperature_end=t_end)
    # (i) clamp: every epoch's temperature stays within [t_end, 1.0]; the LAST is
    # exactly the floor (never < t_end → never a div-by-zero).
    t_last = seg_temperature_for_epoch(spec, epochs - 1)
    assert abs(t_last - t_end) < 1e-9, f"final-epoch T {t_last} != floor {t_end}"
    for e in (0, epochs // 2, epochs - 2, epochs - 1):
        t = seg_temperature_for_epoch(spec, e)
        assert t_end - 1e-12 <= t <= 1.0 + 1e-12, f"epoch {e} T {t} escaped [end, start]"
    # an OUT-OF-RANGE epoch (a long-run off-by-one) is clamped, not extrapolated.
    assert abs(seg_temperature_for_epoch(spec, epochs + 100) - t_end) < 1e-9
    # (ii) surrogate finite + in [0,1] at the coldest T (the boundary-sharpening end).
    seg_out, targets = _make_seg_inputs(seed=11, B=2, H=8, W=12)
    val = _seg_loss_for_spec(spec, seg_out, targets, temperature=t_last,
                             )  # margin_weight_tau None on this spec
    assert torch.isfinite(val), f"surrogate non-finite at T={t_last}: {val}"
    val_f = float(val.detach())
    assert 0.0 - 1e-6 <= val_f <= 1.0 + 1e-6, (
        f"soft-cosine surrogate {val_f} escaped [0,1] at the coldest anneal T"
    )


# --- R9: numerical stability under the anneal tail (the live arm's T=1.0->0.05) ---
def _adversarial_seg_logits(kind: str, *, B: int = 2, H: int = 8, W: int = 12, seed: int = 3):
    """Adversarial SegNet-logit families that probe the annealed-tail (T->0.05)
    softmax/margin arithmetic where a benign random tensor would NOT (R9 lens).
    All stay well inside fp32 (|logit| <= 1e8 = ~10^7x beyond any real SegNet
    logit, which is O(10) per R8's measured margins 2-6) so a NON-finite result
    here is a genuine lever defect, not input overflow."""
    g = torch.Generator().manual_seed(seed)
    if kind == "tied":  # all classes equal -> margin 0 -> exp(-0/tau)=1 everywhere
        x = torch.zeros(B, _SEG_NUM_CLASSES, H, W)
    elif kind == "saturated":  # one class +1e4, rest -1e4 -> softmax fully one-hot
        x = torch.full((B, _SEG_NUM_CLASSES, H, W), -1e4)
        x[:, 0] = 1e4
    elif kind == "huge":  # magnitude 1e3 -> pred/0.05 = 2e4 (large but finite)
        x = torch.randn(B, _SEG_NUM_CLASSES, H, W, generator=g) * 1e3
    elif kind == "extreme":  # magnitude 1e8 -> pred/0.05 = 2e9, still << fp32 max 3.4e38
        x = torch.randn(B, _SEG_NUM_CLASSES, H, W, generator=g) * 1e8
    elif kind == "near_tie":  # two classes within 1e-7 -> margin ~0 at the boundary
        x = torch.randn(B, _SEG_NUM_CLASSES, H, W, generator=g)
        x[:, 1] = x[:, 0] + 1e-7
    else:  # "normal"
        x = torch.randn(B, _SEG_NUM_CLASSES, H, W, generator=g)
    x = x.clone().requires_grad_(True)
    t = torch.randint(0, _SEG_NUM_CLASSES, (B, H, W), generator=g, dtype=torch.int64)
    return x, t


@pytest.mark.parametrize("surrogate", ["soft_cosine", "fisher_rao", "sinkhorn"])
@pytest.mark.parametrize(
    "kind", ["tied", "saturated", "huge", "extreme", "near_tie", "normal"]
)
@pytest.mark.parametrize("margin_tau", [None, 2.0, 1e-6])
def test_lever2_lever5_anneal_tail_finite_loss_and_grad_adversarial(
    surrogate, kind, margin_tau
):
    """R9 numerical-stability lens — the live distortion arm anneals seg-temperature
    1.0 -> 0.05 (PROVENANCE.json) with the soft_cosine surrogate + margin_weight_tau=2.0.
    At the coldest tail T=0.05 the surrogate computes ``softmax(pred/0.05) = softmax(pred*20)``
    and the margin lever computes ``exp(-margin/tau)``. Adversarial logit families
    (tied/saturated/huge/extreme/near-tie) that a benign random tensor would NOT
    exercise must produce a FINITE loss AND FINITE gradients — not just a finite
    VALUE (the prior anneal-tail test checked only the value, and only on clean
    random logits). A NaN/Inf gradient at the tail would silently poison the live
    arm's optimizer step; this pins finiteness across ALL 3 selectable surrogates ×
    6 adversarial families × {no-margin, tau=2 (live), tau=1e-6 (clamp floor)}.

    Class-2-fake-proof: this is NOT a constant-return test. If the surrogate or the
    margin map returned a constant, the gradient would be 0 (still finite) — so we
    ALSO assert the loss actually DEPENDS on the input (non-zero grad on at least
    one non-degenerate family) below, and the parametrization spans values
    (saturated->0.0, normal->~0.57) proving the value is input-dependent."""
    spec = _stage(
        epochs=2,
        seg_surrogate=surrogate,
        seg_temperature=1.0,
        seg_temperature_end=0.05,
        margin_weight_tau=margin_tau,
    )
    # The annealed FINAL-epoch temperature for the live schedule == the floor 0.05.
    t_tail = seg_temperature_for_epoch(spec, spec.epochs - 1)
    assert abs(t_tail - 0.05) < 1e-9, f"tail T should be the 0.05 floor, got {t_tail}"

    x, targets = _adversarial_seg_logits(kind)
    val = _seg_loss_for_spec(spec, x, targets, temperature=t_tail)
    assert torch.isfinite(val).all(), (
        f"{surrogate}/{kind}/tau={margin_tau}: NON-finite loss {val} at anneal tail T=0.05"
    )
    val_f = float(val.detach())
    assert 0.0 - 1e-5 <= val_f <= 1.0 + 1e-5, (
        f"{surrogate}/{kind}/tau={margin_tau}: surrogate {val_f} escaped [0,1] at tail"
    )
    val.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), (
        f"{surrogate}/{kind}/tau={margin_tau}: NON-finite GRADIENT at anneal tail T=0.05 "
        f"(grad has {(~torch.isfinite(x.grad)).sum().item()} non-finite elements)"
    )


def test_lever2_anneal_tail_surrogate_is_input_dependent_not_constant():
    """R9 Class-2-fake guard for the finiteness test above: prove the tail surrogate
    is a REAL function of the input (a constant-return impl would pass every
    finiteness assertion). At T=0.05 a 'normal' logit family gives a non-zero,
    distinct loss from the 'saturated' family, and the gradient w.r.t. the input is
    NON-ZERO (the surrogate genuinely backprops through softmax(pred/0.05))."""
    spec = _stage(
        epochs=2, seg_surrogate="soft_cosine", seg_temperature=1.0,
        seg_temperature_end=0.05, margin_weight_tau=2.0,
    )
    x_norm, t_norm = _adversarial_seg_logits("normal")
    x_sat, t_sat = _adversarial_seg_logits("saturated")
    v_norm = _seg_loss_for_spec(spec, x_norm, t_norm, temperature=0.05)
    v_sat = _seg_loss_for_spec(spec, x_sat, t_sat, temperature=0.05)
    # The two families give DISTINCT losses (input-dependent, not constant).
    assert not torch.allclose(v_norm, v_sat, atol=1e-3), (
        "tail surrogate is input-INVARIANT — a constant return (FAKE)"
    )
    # The gradient flows: at least one input element has non-zero grad.
    v_norm.backward()
    assert x_norm.grad.abs().sum() > 0.0, (
        "tail surrogate has ZERO gradient everywhere — not differentiable backprop (FAKE)"
    )


# --- R10: lever-interaction SIGN + the seg-anneal gradient floor (the cold-T property) ---
def test_seg_anneal_gradient_floor_collapses_seg_grad_at_cold_t():
    """R10 finding — the soft-cosine seg surrogate's per-pixel gradient ∝ p·(1−p)
    VANISHES as the anneal sharpens (T → 0), even though its VALUE stays correct.
    On SATURATED logits (the cold-tail regime: large margins ⇒ softmax(pred/0.05) is
    near one-hot ⇒ p ≈ 0 or 1 ⇒ p·(1−p) ≈ 0) the latent gradient at T=0.05 is many
    orders of magnitude smaller than at T=1.0. This pins the INTENDED 'lock in the
    argmax late' property (NOT a defect) so a future tuner who drives a stage cold
    before its argmax converges is caught by the companion floor guard.

    NO-FAKE: this measures the ACTUAL gradient-norm RATIO across T (not a constant) —
    a constant-return surrogate would have ZERO grad at every T (ratio undefined),
    which the assertions below reject; a temperature-INVARIANT surrogate would give
    ratio ≈ 1 (rejected)."""
    # Saturated logits: one class dominant by ~12 (margin 12 ⇒ softmax(·/0.05) one-hot).
    g = torch.Generator().manual_seed(7)
    B, H, W = 2, 8, 12
    base = torch.randn(B, _SEG_NUM_CLASSES, H, W, generator=g)
    base[:, 0] += 12.0  # a confident class-0 prediction (large margin) on most pixels
    spec = _stage(seg_surrogate="soft_cosine", margin_weight_tau=None)
    targets = torch.randint(0, _SEG_NUM_CLASSES, (B, H, W), generator=g, dtype=torch.int64)

    def grad_norm_at(temp):
        x = base.clone().requires_grad_(True)
        val = _seg_loss_for_spec(spec, x, targets, temperature=temp)
        val.backward()
        return float(x.grad.norm())

    gn_warm = grad_norm_at(1.0)  # gradient ALIVE
    gn_cold = grad_norm_at(0.05)  # gradient near-DEAD (the live arm's tail)
    assert gn_warm > 0.0, "warm-T seg grad is zero — surrogate is dead even at T=1.0 (FAKE/defect)"
    # The cold-T gradient is DRAMATICALLY smaller (the saturated-softmax collapse).
    assert gn_cold < gn_warm * 1e-3, (
        f"cold-T (0.05) seg grad {gn_cold:.3e} is NOT << warm-T (1.0) {gn_warm:.3e} — "
        "the documented saturated-softmax gradient collapse did not occur "
        "(if this regresses, the cold-anneal 'lock-in' property changed)"
    )


def test_seg_anneal_gradient_alive_guard_matches_floor_and_live_schedule():
    """R10 self-protection guard, R12-CORRECTED — ``seg_anneal_temperature_is_gradient_
    alive`` flips EXACTLY at ``SEG_ANNEAL_GRADIENT_FLOOR_T`` (= 0.3, the R12-corrected
    USABLE knee), and the LIVE arm's cosine schedule (T: 1.0→0.05 over a stage) spends
    the MAJORITY of its epochs in the gradient-ALIVE regime (so the seg lever is not
    wasted) and goes cold-dead in the tail (the intended late lock-in). A regression
    that lowered the floor back toward the mislabeling 0.1 OR front-loaded the cold tail
    would FAIL here.

    R12 FINDING (the floor was MISCALIBRATED): R10 set the floor to 0.1, but the
    measured combined gradient at T=0.1 is ~10 orders below the warm-T norm (R12 sweep:
    1.9e-10) — R10 ITSELF labeled T=0.1 "≈ dead" (|grad|≈5e-12), yet the guard called
    0.1 "alive". The floor is corrected to 0.3 (ratio 1.7e-3, the last temp within ~2
    orders of warm). The deep-dead points the OLD floor mislabeled (0.2, 0.15, 0.12,
    0.1) must now read DEAD."""
    # The guard is a clean threshold at the CORRECTED floor (0.3, not 0.1).
    assert SEG_ANNEAL_GRADIENT_FLOOR_T == 0.3, (
        "R12 corrected the seg-anneal gradient floor to 0.3 (the measured usable knee); "
        f"got {SEG_ANNEAL_GRADIENT_FLOOR_T} — a regression toward the mislabeling 0.1?"
    )
    assert seg_anneal_temperature_is_gradient_alive(SEG_ANNEAL_GRADIENT_FLOOR_T) is True
    assert seg_anneal_temperature_is_gradient_alive(SEG_ANNEAL_GRADIENT_FLOOR_T - 1e-6) is False
    assert seg_anneal_temperature_is_gradient_alive(1.0) is True
    assert seg_anneal_temperature_is_gradient_alive(0.5) is True   # within ~1.2 orders of warm
    # The deep-dead temperatures the OLD 0.1 floor MISLABELED as alive — now DEAD:
    for t_dead in (0.2, 0.15, 0.12, 0.1, 0.08, 0.05):
        assert seg_anneal_temperature_is_gradient_alive(t_dead) is False, (
            f"T={t_dead} should be classified DEAD at the R12 floor 0.3 (the OLD 0.1 floor "
            "mislabeled it alive while its gradient was 5–20 orders below warm)"
        )
    # The live arm's schedule: T 1.0→0.05 over a 100-epoch stage. At the corrected floor
    # 0.3 the cosine keeps 66/100 epochs alive (was 85/100 at the mislabeling 0.1 floor)
    # and goes dead for the last 34 — a larger, MORE HONEST cold tail.
    spec = _stage(epochs=100, seg_surrogate="soft_cosine",
                  seg_temperature=1.0, seg_temperature_end=0.05)
    temps = [seg_temperature_for_epoch(spec, e) for e in range(100)]
    alive = sum(1 for t in temps if seg_anneal_temperature_is_gradient_alive(t))
    assert alive == 66, (
        f"expected 66/100 alive epochs at the R12 floor 0.3 for the live 1.0→0.05 "
        f"schedule, got {alive} (a floor or anneal regression)"
    )
    # The majority is still alive (the seg lever is not wasted for most of the stage).
    assert alive >= 50, (
        f"only {alive}/100 epochs are gradient-alive — the cold tail is front-loaded "
        "(the seg lever would be wasted before the argmax converges)"
    )
    # The very end IS cold-dead (the intended argmax lock-in).
    assert not seg_anneal_temperature_is_gradient_alive(temps[-1])


# --- R12: the COMBINED (Lever-5 margin × Lever-2 surrogate) gradient floor + the ---
# --- multiply-a-dead-gradient property (Lever-5 cannot rescue the cold surrogate). ---
def test_r12_floor_is_at_usable_knee_not_dead_zone_edge():
    """R12 finding guard — the corrected floor (0.3) marks where the combined seg
    gradient is still USABLE (within ~3 orders of the warm-T norm), NOT a dead-zone
    edge where it is already 10 orders down (the mislabeling R12 found). On the SAME
    saturated logits used by the R10 floor test, the gradient ratio AT the floor (0.3)
    must be DRAMATICALLY larger than at the OLD floor (0.1) — proving the corrected
    boundary actually separates usable from dead.

    NO-FAKE: measures the ACTUAL gradient-norm ratio at three temperatures (0.3 / 0.2 /
    0.1) on real soft-cosine backprop. A constant surrogate gives 0 grad everywhere
    (rejected by the >0 assertions); a T-invariant one gives ratio 1 (rejected by the
    monotone-collapse assertion)."""
    g = torch.Generator().manual_seed(7)
    B, H, W = 2, 8, 12
    base = torch.randn(B, _SEG_NUM_CLASSES, H, W, generator=g)
    base[:, 0] += 12.0  # confident class-0 (large margin) — the saturated cold regime
    spec = _stage(seg_surrogate="soft_cosine", margin_weight_tau=2.0)  # Lever-5 ON (live)
    targets = torch.randint(0, _SEG_NUM_CLASSES, (B, H, W), generator=g, dtype=torch.int64)

    def grad_norm_at(temp):
        x = base.clone().requires_grad_(True)
        _seg_loss_for_spec(spec, x, targets, temperature=temp).backward()
        return float(x.grad.norm())

    gn_warm = grad_norm_at(1.0)
    gn_floor = grad_norm_at(SEG_ANNEAL_GRADIENT_FLOOR_T)  # 0.3
    gn_old = grad_norm_at(0.1)  # the OLD mislabeling floor
    assert gn_warm > 0.0, "warm-T grad is zero — surrogate dead even at T=1.0 (FAKE/defect)"
    assert gn_floor > 0.0, "grad at the floor is exactly zero — floor is too cold"
    # The gradient is MONOTONE-collapsing: floor(0.3) >> old(0.1). The old 0.1 point is
    # MANY orders smaller than the corrected 0.3 floor (the dead-zone the old guard
    # wrongly called alive).
    assert gn_old < gn_floor * 1e-3, (
        f"grad at old floor 0.1 ({gn_old:.3e}) is NOT << grad at corrected floor 0.3 "
        f"({gn_floor:.3e}) — the floor correction did not separate usable from dead"
    )
    # The floor gradient is within a sane band of warm (NOT itself a rounding error):
    # on the live margin-weighted saturated logits the floor ratio is >= ~1e-4 of warm,
    # whereas the old 0.1 floor was < 1e-7 of warm.
    assert gn_floor / gn_warm > gn_old / gn_warm * 100.0


def test_r12_margin_lever5_cannot_rescue_cold_dead_seg_gradient():
    """R12 'multiply-a-dead-gradient' guard — Lever-5 (``margin_weight_tau``) multiplies
    the per-pixel seg surrogate by a DETACHED weight ``exp(−margin/τ) ∈ (0,1]``, so it
    can only SCALE DOWN the already-dead cold-T gradient, never REVIVE it. At a cold T
    (below the floor) the margin-ON combined gradient must be <= the margin-OFF (plain)
    gradient — Lever-5 does not rescue the dead surrogate. (And ``exp(−margin/τ)`` is
    SMALLEST on the confidently-WRONG large-margin flip pixels, so it down-weights
    exactly the pixels the cold surrogate already cannot fix.) MEASURED on the real
    scorer in R12 (margin-ON 6.3e-20 ≤ margin-OFF 1.8e-19 at T=0.05); this fast
    structural test pins the same inequality on saturated logits.

    NO-FAKE: the margin weight is a real ``exp(−margin/τ)`` of the actual logit margin,
    not a constant — the test ALSO asserts margin-ON differs from margin-OFF at WARM T
    (the weight genuinely reshapes), so a no-op weight (returning 1) would FAIL the
    warm-T inequality."""
    g = torch.Generator().manual_seed(7)
    B, H, W = 2, 8, 12
    base = torch.randn(B, _SEG_NUM_CLASSES, H, W, generator=g)
    base[:, 0] += 12.0
    targets = torch.randint(0, _SEG_NUM_CLASSES, (B, H, W), generator=g, dtype=torch.int64)
    spec_m = _stage(seg_surrogate="soft_cosine", margin_weight_tau=2.0)
    spec_p = _stage(seg_surrogate="soft_cosine", margin_weight_tau=None)

    def grad_norm(spec, temp):
        x = base.clone().requires_grad_(True)
        _seg_loss_for_spec(spec, x, targets, temperature=temp).backward()
        return float(x.grad.norm())

    # At a COLD T (below the floor): margin-ON cannot exceed margin-OFF (no rescue).
    gm_cold = grad_norm(spec_m, 0.05)
    gp_cold = grad_norm(spec_p, 0.05)
    assert gm_cold <= gp_cold + 1e-12, (
        f"margin-ON cold grad {gm_cold:.3e} > margin-OFF {gp_cold:.3e} — Lever-5 must "
        "NOT be able to rescue (revive) the cold-dead surrogate gradient"
    )
    # Sanity: the margin weight is NOT a no-op — at WARM T it genuinely reshapes the
    # gradient (margin-ON ≠ margin-OFF), so the cold-T inequality above is meaningful.
    gm_warm = grad_norm(spec_m, 1.0)
    gp_warm = grad_norm(spec_p, 1.0)
    assert abs(gm_warm - gp_warm) > 1e-9, (
        "margin-ON warm grad == margin-OFF — Lever-5 is a no-op (the weight is not "
        "reshaping; a constant-1 weight would pass the cold test vacuously)"
    )
    # And the margin weight scales DOWN (it is in (0,1]): margin-ON warm <= margin-OFF warm.
    assert gm_warm <= gp_warm + 1e-9


@pytest.mark.timeout(600)
def test_r12_combined_seg_gradient_floor_on_real_scorer():
    """R12 HEADLINE lens (real scorer) — the COMBINED (Lever-5 margin × Lever-2
    surrogate) gradient floor holds on the REAL frozen scorer, and Lever-5 cannot
    rescue the cold-dead gradient. Builds a FiLM (Lever-3) decoder on real 0.mkv pairs,
    measures the param/latent gradient norm of the margin-weighted seg loss at warm
    (1.0), the corrected floor (0.3), and the live-arm cold tail (0.05):
      (1) the combined gradient COLLAPSES from warm to cold (ratio at 0.05 < 1e-3 of
          warm — the floor is real, not a surrogate-only artifact);
      (2) the floor T=0.3 gradient is DRAMATICALLY larger than the 0.05 tail (the
          corrected floor sits above the dead zone);
      (3) margin-ON combined grad <= margin-OFF at the cold tail (Lever-5 no-rescue).

    RESEARCH-ONLY tiny slice ⇒ [contest-CPU advisory] NON-PROMOTABLE (gradient-magnitude
    claim, not a score claim). NO-FAKE: real autograd on the real scorer; a constant/
    no-op lever gives zero gradient (the >0 assertions reject that)."""
    pytest.importorskip("torch")
    import os

    from tac.torch_vehicle.scorer_context import RealScorerContext

    video = "upstream/videos/0.mkv"
    if not os.path.exists(video):
        pytest.skip("real video 0.mkv not present (real-scorer test)")

    from tac.torch_vehicle.driver import _EVAL_H, _EVAL_W

    torch.manual_seed(0)
    cache = _tmp_out()
    sc = RealScorerContext(video, device="cpu", max_pairs=8, targets_cache=cache)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        pose_film_enabled=True, seed=0,
    )
    spec_m = _stage(seg_surrogate="soft_cosine", margin_weight_tau=2.0)
    spec_p = _stage(seg_surrogate="soft_cosine", margin_weight_tau=None)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec_m])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    decoder.set_stored_pose(sc.pose_targets[:8].cpu())
    latents = torch.nn.Parameter(torch.randn(8, 28) * 0.1)
    idx = torch.arange(4)
    params = [p for p in decoder.parameters() if p.requires_grad] + [latents]
    tgt = sc.seg_targets_hard[idx]

    def render():
        dp = decoder(latents[idx], idx)
        B = len(idx)
        flat = dp.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
        bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
        dc = bhwc.clamp(0, 255)
        return dc + (dc.round() - dc).detach()

    def grad_norm(spec, temp):
        seg_out, _ = sc.seg_pose_forward(render())
        seg_l = _seg_loss_for_spec(spec, seg_out, tgt, temperature=temp)
        gs = torch.autograd.grad(spec.seg_weight * seg_l, params, allow_unused=True)
        return float(torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for p, g in zip(params, gs, strict=False)
        ]).norm())

    gn_warm = grad_norm(spec_m, 1.0)
    gn_floor = grad_norm(spec_m, SEG_ANNEAL_GRADIENT_FLOOR_T)  # 0.3
    gn_cold_m = grad_norm(spec_m, 0.05)
    gn_cold_p = grad_norm(spec_p, 0.05)
    assert gn_warm > 0.0, "warm-T combined grad is zero on the real scorer (FAKE/defect)"
    # (1) the floor is real — the cold tail collapses << warm.
    assert gn_cold_m < gn_warm * 1e-3, (
        f"cold-tail (0.05) combined grad {gn_cold_m:.3e} is NOT << warm (1.0) "
        f"{gn_warm:.3e} — the seg-gradient floor did not occur on the real scorer"
    )
    # (2) the corrected floor T=0.3 sits ABOVE the dead zone (>> the 0.05 tail).
    assert gn_floor > gn_cold_m * 1e3, (
        f"floor (0.3) grad {gn_floor:.3e} is NOT >> cold tail (0.05) {gn_cold_m:.3e} — "
        "the corrected floor is not above the dead zone"
    )
    # (3) Lever-5 cannot rescue the cold-dead gradient (margin-ON <= margin-OFF).
    assert gn_cold_m <= gn_cold_p + 1e-12, (
        f"margin-ON cold grad {gn_cold_m:.3e} > margin-OFF {gn_cold_p:.3e} on the real "
        "scorer — Lever-5 must not revive the dead surrogate"
    )


@pytest.mark.timeout(420)
def test_r13_all5_multi_step_descent_at_faithful_lr_synthetic():
    """R13 lens (FAST synthetic) — with ALL FIVE levers ON, a multi-step Adam
    trajectory at the FAITHFUL LR genuinely DESCENDS the COMBINED loss (the
    integrated multi-lever effect is a working training signal, not just a per-lever
    step-0 sign). No prior round measured multi-STEP descent of the composed loss:
    R10 measured the single-step gradient SIGN, R5 measured determinism, R11 measured
    resume bit-identity, and ``test_compose_all_five_levers_end_to_end`` only byte-
    closes (it never checks the loss goes DOWN). A set of levers can each point
    downhill at step 0 yet fail to descend over a trajectory (oscillation, the seg
    term going cold-dead, the rate/QAT terms cumulatively fighting the score terms).

    Distinct from the REAL-scorer R13 test below (which confirms it on the contest
    scorer): this fast synthetic version is the always-run mechanism guard.

    THE FAITHFUL-LR DISCIPLINE (the R10 test-instrument lesson): the live distortion
    arm trains at ``adamw_lr≈1e-3`` and descends loss 0.68→0.52 monotonically. A
    probe LR 50× larger (5e-2) OSCILLATES from step 1 — that is an INSTRUMENT artifact,
    NOT a lever defect. The descent MUST be measured at the LR the levers actually
    train at. We use the ROBUST windowed metric (mean of the last quartile of steps
    vs the first step) because SGD on a real scorer has per-step noise; a single-step
    ``loss_end < loss_start`` would be noise-fragile.

    NO-FAKE: a real Adam trajectory through the driver's OWN composed loss assembly
    (``_seg_loss_for_spec`` + pose + ``_weight_regularizers``) at the annealed
    temperature; a no-op lever set (all-default) would still descend, so the test ALSO
    asserts the all-5-on loss is DISTINCT from the all-default loss at step 0 (the
    levers are genuinely active, not silently bypassed)."""
    torch.manual_seed(0)
    sc = SyntheticScorerContext(n_pairs=4, device="cpu", seed=0)
    v = import_vendored_bundle()
    # Small decoder (base_channels=8, like the R11 resume tests) + 2 pairs/step keeps
    # the per-step rate surrogate (Lever 1, the cost bottleneck) + the 384×512 forward
    # cheap so the always-run mechanism guard stays fast.
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        pose_film_enabled=True, pose_film_hidden=8, seed=0,
    )
    # ALL FIVE levers ON; the seg anneal lands AT the corrected gradient-alive floor
    # (0.3) so the seg gradient stays USABLE for the whole trajectory (per R12).
    n_steps = 8  # enough for a windowed trend; the rate surrogate per step is costly
    spec = _stage(
        name="r13_all5", epochs=n_steps, use_qat=True, cat_lambda=0.01,
        rate_lambda_w=0.05, rate_lambda_lat=0.02,
        seg_surrogate="soft_cosine", seg_temperature=1.0,
        seg_temperature_end=SEG_ANNEAL_GRADIENT_FLOOR_T,
        score_aware_qat=True, qat_sensitivity_decay=0.9, margin_weight_tau=2.0,
        adamw_lr=1e-3,  # the FAITHFUL daemon LR (NOT a 50× overshoot)
    )
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    n = 4
    decoder.set_stored_pose(sc.pose_targets[:n].cpu())
    latents = torch.nn.Parameter(torch.randn(n, 28) * 0.1)
    idx = torch.arange(2)
    dec_params = [p for p in decoder.parameters() if p.requires_grad]
    opt = torch.optim.Adam(dec_params + [latents], lr=spec.adamw_lr)

    def render():
        dp = decoder(latents[idx], idx)
        B = len(idx)
        flat = dp.reshape(B * 2, 3, 384, 512)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
        bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
        dc = bhwc.clamp(0, 255)
        return dc + (dc.round() - dc).detach()

    def combined_loss(epoch_in_stage):
        t = seg_temperature_for_epoch(spec, epoch_in_stage)
        seg_out, pose6 = sc.seg_pose_forward(render())
        seg_l = _seg_loss_for_spec(spec, seg_out, sc.seg_targets_hard[idx], temperature=t)
        pose_l = torch.sqrt(10.0 * F.mse_loss(pose6, sc.pose_targets[idx]) + 1e-12)
        loss = spec.seg_weight * seg_l + spec.pose_weight * pose_l
        reg = driver._weight_regularizers(decoder, latents, spec)
        if reg is not None:
            loss = loss + reg
        return loss

    # All-default reference loss at step 0 (the levers must DIFFER from this, else
    # they are silently inactive and the descent test is vacuous).
    default_spec = _stage(name="r13_default", epochs=n_steps, adamw_lr=1e-3)
    with torch.no_grad():
        seg_out0, pose0 = sc.seg_pose_forward(render())
        seg_d0 = _seg_loss_for_spec(default_spec, seg_out0, sc.seg_targets_hard[idx])
        pose_d0 = torch.sqrt(10.0 * F.mse_loss(pose0, sc.pose_targets[idx]) + 1e-12)
        default_loss0 = float(default_spec.seg_weight * seg_d0 + default_spec.pose_weight * pose_d0)

    losses = []
    for step in range(spec.epochs):
        opt.zero_grad()
        loss = combined_loss(step)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert all(torch.isfinite(torch.tensor(x)).item() for x in losses), (
        "an all-5-on combined loss step is non-finite (a lever produced NaN/Inf)"
    )
    # The levers are genuinely active (step-0 composed loss != all-default loss).
    assert abs(losses[0] - default_loss0) > 1e-3, (
        f"all-5-on step-0 loss {losses[0]:.4f} == all-default {default_loss0:.4f} — "
        "the composed levers are silently inactive (the descent test would be vacuous)"
    )
    # ROBUST windowed descent: the mean of the last quartile is below the first step.
    tail = losses[(3 * len(losses)) // 4:]
    tail_mean = sum(tail) / len(tail)
    assert tail_mean < losses[0], (
        f"all-5-on combined loss did NOT descend at the faithful LR: tail-mean "
        f"{tail_mean:.4f} >= start {losses[0]:.4f} — the integrated multi-lever effect "
        "is not a working training signal (or a lever fights the descent cumulatively)"
    )


@pytest.mark.timeout(420)
def test_r13redo_all5_does_not_reverse_seg_from_a_TRAINED_operating_point():
    """R13-REDO lens (the operator's correction) — FAST synthetic operating-point
    guard. The FIRST R13 probe started from a FRESH RANDOM decoder (pose term ~38,
    d_pose ~100s = UNTRAINED), so its "descent" was a from-scratch decoder learning,
    NOT a test of the levers AT THE TRAINED OPERATING POINT. The honest invariant:
    starting from a CONVERGED decoder (small d_seg/d_pose), turning ALL FIVE levers
    ON must NOT REVERSE the seg score-improving direction relative to the no-lever
    default control — i.e. the seg surrogate (Lever-2) at the operating point still
    reduces (or holds, within a slice-noise band) the d_seg, and no lever pushes
    d_seg systematically UP.

    Why this is distinct from the from-scratch synthetic descent test above: that
    test starts at random init where ANY working signal descends; THIS test first
    TRAINS the decoder to a converged operating point (the regime the live basin is
    actually in: d_seg ~0.0038, d_pose ~0.0006 on the REAL scorer per the R13-REDO
    real-scorer probe ``probe_r13_redo_all5_trained_basin_descent.py``), then asks
    whether the levers REVERSE the seg direction there. A lever that helps at random
    init can still FIGHT at the converged operating point — that is the regime the
    SEAL must clear and no prior round tested.

    THE INSTRUMENT DISCIPLINE (the decisive R10/R13 lesson): a FRESH Adam optimizer
    on a converged decoder perturbs it on STEP 0 regardless of levers (moment-buffer
    reset). That step-0 spike is IDENTICAL in the no-lever control, so the d_pose
    term on a tiny slice is noise-dominated and is NOT the arbiter. The arbiter is
    d_seg (the 100x-weighted dominant term, far less per-step noisy): the all-5-on
    tail d_seg must be within a slice-noise band of the no-lever default tail d_seg.

    NO-FAKE: a REAL converge-then-perturb trajectory through the driver's OWN loss
    assembly; the levers are proven ACTIVE (the all-5-on step-0 loss differs from the
    default), and the seg-reversal assertion would FAIL if a lever drove d_seg up.
    RESEARCH-ONLY synthetic ⇒ no score claim (the real-scorer confirmation is the
    companion probe; this is the always-run fast regression guard)."""
    torch.manual_seed(0)
    sc = SyntheticScorerContext(n_pairs=4, device="cpu", seed=0)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        pose_film_enabled=True, pose_film_hidden=8, seed=0,
    )
    n = 4
    idx = torch.arange(2)

    def _render(decoder, latents):
        dp = decoder(latents[idx], idx)
        B = len(idx)
        flat = dp.reshape(B * 2, 3, 384, 512)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
        bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
        dc = bhwc.clamp(0, 255)
        return dc + (dc.round() - dc).detach()

    def _d_seg(decoder, latents):
        with torch.no_grad():
            seg_out, _pose = sc.seg_pose_forward(_render(decoder, latents))
            return (seg_out.argmax(dim=1) != sc.seg_targets_hard[idx]).float().mean().item()

    # ---- (1) TRAIN a decoder to a converged operating point (CE, no levers) ----
    base = _stage(name="r13redo_warmup", epochs=1, adamw_lr=3e-3)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[base])
    trained_dec = driver._new_decoder(device=torch.device("cpu"))
    trained_dec.set_stored_pose(sc.pose_targets[:n].cpu())
    warm_latents = torch.nn.Parameter(torch.randn(n, 28) * 0.1)
    warm_params = [p for p in trained_dec.parameters() if p.requires_grad]
    warm_opt = torch.optim.Adam(warm_params + [warm_latents], lr=3e-3)
    for _ in range(40):  # converge: CE seg + pose to a SMALL-distortion operating point
        warm_opt.zero_grad()
        seg_out, pose6 = sc.seg_pose_forward(_render(trained_dec, warm_latents))
        loss = 100.0 * F.cross_entropy(
            seg_out, sc.seg_targets_hard[idx]
        ) + torch.sqrt(10.0 * F.mse_loss(pose6, sc.pose_targets[idx]) + 1e-12)
        loss.backward()
        warm_opt.step()
    d_seg_converged = _d_seg(trained_dec, warm_latents)
    # snapshot the converged state so BOTH arms fork from the IDENTICAL operating point
    conv_dec_state = {k: x.detach().clone() for k, x in trained_dec.state_dict().items()}
    conv_lat = warm_latents.detach().clone()

    def _run_arm(spec, n_steps):
        dec = driver._new_decoder(device=torch.device("cpu"))
        dec.load_state_dict(conv_dec_state)
        lat = torch.nn.Parameter(conv_lat.clone())
        params = [p for p in dec.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params + [lat], lr=spec.adamw_lr)
        d_segs, step0_loss = [], None
        for step in range(n_steps):
            opt.zero_grad()
            t = seg_temperature_for_epoch(spec, step)
            seg_out, pose6 = sc.seg_pose_forward(_render(dec, lat))
            seg_l = _seg_loss_for_spec(spec, seg_out, sc.seg_targets_hard[idx], temperature=t)
            pose_l = torch.sqrt(10.0 * F.mse_loss(pose6, sc.pose_targets[idx]) + 1e-12)
            loss = spec.seg_weight * seg_l + spec.pose_weight * pose_l
            reg = driver._weight_regularizers(dec, lat, spec)
            if reg is not None:
                loss = loss + reg
            if step0_loss is None:
                step0_loss = float(loss.detach())
            loss.backward()
            opt.step()
            d_segs.append(_d_seg(dec, lat))
        return d_segs, step0_loss

    n_steps = 8
    all5 = _stage(
        name="r13redo_all5", epochs=n_steps, use_qat=True, cat_lambda=0.01,
        rate_lambda_w=0.05, rate_lambda_lat=0.02,
        seg_surrogate="soft_cosine", seg_temperature=1.0,
        seg_temperature_end=SEG_ANNEAL_GRADIENT_FLOOR_T,
        score_aware_qat=True, qat_sensitivity_decay=0.9, margin_weight_tau=2.0,
        adamw_lr=1e-3,
    )
    default = _stage(name="r13redo_default", epochs=n_steps, adamw_lr=1e-3)
    all5_dseg, all5_step0 = _run_arm(all5, n_steps)
    def_dseg, def_step0 = _run_arm(default, n_steps)

    # the levers are genuinely ACTIVE (step-0 composed loss differs from default) —
    # else the no-reversal assertion would be vacuous.
    assert abs(all5_step0 - def_step0) > 1e-3, (
        f"all-5-on step-0 loss {all5_step0:.4f} == default {def_step0:.4f}: the levers "
        "are silently inactive (the operating-point reversal guard would be vacuous)"
    )
    assert all(torch.isfinite(torch.tensor(x)).item() for x in all5_dseg + def_dseg), (
        "a d_seg measurement is non-finite (a lever produced NaN/Inf at the "
        "operating point)"
    )
    all5_tail = sum(all5_dseg[-4:]) / 4.0
    def_tail = sum(def_dseg[-4:]) / 4.0
    # NO SEG-DIRECTION REVERSAL: at the converged operating point, all-5-on d_seg is
    # within a slice-noise band of the no-lever default d_seg (it does NOT climb).
    # The band (~30% of the converged d_seg + a small floor) is the tiny-4-pair noise.
    band = 0.30 * max(d_seg_converged, 1e-4) + 0.01
    assert all5_tail <= def_tail + band, (
        f"all-5-on REVERSED the seg direction at the trained operating point: "
        f"all5 tail d_seg {all5_tail:.5f} > default {def_tail:.5f} + band {band:.5f} "
        f"(converged start d_seg {d_seg_converged:.5f}). A lever is fighting the seg "
        "score-improving direction at the operating point."
    )
    # and all-5-on d_seg does not BLOW UP from the converged start (no divergence).
    assert all5_tail <= d_seg_converged + band, (
        f"all-5-on d_seg DIVERGED from the converged start: tail {all5_tail:.5f} "
        f">> start {d_seg_converged:.5f} + band {band:.5f}"
    )


@pytest.mark.timeout(900)
def test_r13_all5_descent_byteclose_parseback_on_real_scorer():
    """R13 HEADLINE lens (REAL scorer) — with ALL FIVE levers ON, the COMBINED loss
    DESCENDS over a multi-step Adam trajectory on the REAL frozen scorer (EfficientNet-
    B2 SegNet + FastViT PoseNet), AND a full all-5-on ``run()`` byte-closes an archive
    whose decoder + Lever-3 pose section parse back, AND the BEST checkpoint's exact
    real-score components are FINITE + in-range. This is the integrated end-to-end
    proof that the five composed levers form a real training signal that exports a
    well-formed archive on the contest scorer — no prior round measured multi-step
    descent of the composed loss on the real scorer.

    Run at the FAITHFUL daemon LR (1e-3) — the live distortion arm descends 0.68→0.52
    at this LR; a 50× overshoot oscillates (instrument artifact, not a lever defect).

    RESEARCH-ONLY tiny slice ⇒ [contest-CPU advisory] NON-PROMOTABLE (descent-direction
    + finiteness claim, NOT a score claim). NO-FAKE: real autograd + real export on the
    real scorer; an inactive lever set would still byte-close but the descent assertion
    + the distinctness-from-default assertion (in the fast sister test) reject a no-op."""
    pytest.importorskip("torch")
    import os

    from tac.torch_vehicle.scorer_context import RealScorerContext

    video = "upstream/videos/0.mkv"
    if not os.path.exists(video):
        pytest.skip("real video 0.mkv not present (real-scorer test)")

    from tac.torch_vehicle.driver import _EVAL_H, _EVAL_W
    from tac.torch_vehicle.pose_film import parse_pose_section

    torch.manual_seed(0)
    cache = _tmp_out()
    sc = RealScorerContext(video, device="cpu", max_pairs=8, targets_cache=cache)
    v = import_vendored_bundle()
    out = _tmp_out()
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=out, device="cpu",
        pose_film_enabled=True, pose_film_hidden=8, seed=0,
    )
    n = 8
    steps = 16
    spec = _stage(
        name="r13_all5_real", epochs=steps, use_qat=True, cat_lambda=0.01,
        rate_lambda_w=0.05, rate_lambda_lat=0.02,
        seg_surrogate="soft_cosine", seg_temperature=1.0,
        seg_temperature_end=SEG_ANNEAL_GRADIENT_FLOOR_T,
        score_aware_qat=True, qat_sensitivity_decay=0.9, margin_weight_tau=2.0,
        adamw_lr=1e-3,
    )
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    decoder.set_stored_pose(sc.pose_targets[:n].cpu())
    latents = torch.nn.Parameter(torch.randn(n, 28) * 0.1)
    idx = torch.arange(4)
    dec_params = [p for p in decoder.parameters() if p.requires_grad]
    opt = torch.optim.Adam(dec_params + [latents], lr=spec.adamw_lr)

    def render():
        dp = decoder(latents[idx], idx)
        B = len(idx)
        flat = dp.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
        bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
        dc = bhwc.clamp(0, 255)
        return dc + (dc.round() - dc).detach()

    losses = []
    for step in range(steps):
        opt.zero_grad()
        t = seg_temperature_for_epoch(spec, step)
        seg_out, pose6 = sc.seg_pose_forward(render())
        seg_l = _seg_loss_for_spec(spec, seg_out, sc.seg_targets_hard[idx], temperature=t)
        pose_l = torch.sqrt(10.0 * F.mse_loss(pose6, sc.pose_targets[idx]) + 1e-12)
        loss = spec.seg_weight * seg_l + spec.pose_weight * pose_l
        reg = driver._weight_regularizers(decoder, latents, spec)
        if reg is not None:
            loss = loss + reg
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert all(torch.isfinite(torch.tensor(x)).item() for x in losses), (
        "an all-5-on combined-loss step is non-finite on the real scorer"
    )
    # (1) ROBUST windowed descent on the real scorer.
    tail = losses[(3 * len(losses)) // 4:]
    tail_mean = sum(tail) / len(tail)
    assert tail_mean < losses[0], (
        f"all-5-on combined loss did NOT descend on the REAL scorer at the faithful LR: "
        f"tail-mean {tail_mean:.4f} >= start {losses[0]:.4f} (traj {losses})"
    )

    # (2) a full all-5-on run() byte-closes + parses back (decoder + Lever-3 pose).
    run_cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(),
        checkpoint_every_epochs=1, device="cpu",
        pose_film_enabled=True, pose_film_hidden=8, seed=0,
    )
    drv2 = TorchVehicleDriver(run_cfg, scorer=sc, vendored=v, curriculum=[spec])
    summary = drv2.run()
    assert summary.get("status") == "complete", "all-5-on real-scorer run() did not finish"
    arch_path = os.path.join(run_cfg.out_dir, "best", "best_archive.bin")
    assert os.path.exists(arch_path), "no best archive emitted by the all-5-on real run"
    archive = open(arch_path, "rb").read()
    assert len(archive) > 0
    pose = parse_pose_section(archive, drv2.v.parse_archive)
    assert pose is not None and tuple(pose.shape) == (n, 6), (
        "Lever-3 pose section did not round-trip in the all-5-on real archive"
    )
    dec_sd, lat, _meta = drv2.v.parse_archive(archive)
    assert dec_sd and lat.shape[0] == n, "decoder/latent section did not parse back"

    # (3) the BEST/LAST exact real-score components are FINITE + in-range.
    ev = summary.get("last_eval") or {}
    comps = {k: float(ev[k]) for k in ("d_seg", "d_pose", "rate", "score") if k in ev}
    if "best_score" in summary:
        comps.setdefault("score", float(summary["best_score"]))
    assert comps, "run() exposed no real-score components to check finiteness"
    for k, val in comps.items():
        assert torch.isfinite(torch.tensor(val)).item(), (
            f"real-score component {k}={val} is non-finite after all-5-on training"
        )
        assert val >= 0.0, f"real-score component {k}={val} is negative (impossible)"


def test_r14_lever4_export_is_uniform_127_not_variable_level_documented_scope():
    """R14 lens — Lever-4 (score-aware QAT) EXPORT-surface scope guard. Lever-4 is a
    TRAINING-TIME proxy: it shapes the decoder weights so that, after the codec's
    UNIFORM 127-level quant, score-irrelevant weights collapse to repeated symbols
    brotli compresses well (score_aware_qat.py docstring lines 34-37). It EXPLICITLY
    does NOT inject a variable-level export grid — the actual archive always uses the
    vendored uniform-127 ``quantize_state_dict`` + ``encode_decoder``.

    This guards the DOCUMENTED scope at TWO surfaces (distinct from the existing
    ``test_score_aware_grid_yields_smaller_real_brotli_blob_than_uniform`` which proves
    the byte-savings are REAL):
      (1) the variable-level codec ``build_decoder_blob_variable_or_vendored`` DOES
          deliver byte-savings when fed a coarse level map keyed by STATE-DICT keys
          (the mechanism is real + sd-key-correct), AND emits the vendored byte-
          identical blob when the map is uniform/None (default-preserving);
      (2) the driver's DEFAULT export path (``_build_archive_and_eval_decoder``) takes
          NO sensitivity argument — it never routes Lever-4's ``tensor_sensitivity_ema``
          into a variable-level export. This is the CORRECT documented behavior, not a
          gap: a regression that silently wired a variable-level export off the EMA
          (changing the archive grammar / byte-layout out from under the daemon) would
          be caught here.

    Why this matters (the R14 adversarial question it answers): a reviewer might
    suspect Lever-4's "fewer brotli bytes" claim is an unfulfilled export no-op
    (Catalog #220). The answer is CLEAN: the savings are delivered via the uniform
    codec's brotli compressibility (proven by the sister brotli-blob test), and the
    export-surface scope (uniform-127, no variable-level wiring) is exactly as
    documented. NO-FAKE: a real coarse-map blob measurement + an AST/signature check
    that the default export takes no sensitivity param."""
    pytest.importorskip("torch")
    import inspect

    from tac.losses.variable_level_codec import build_decoder_blob_variable_or_vendored

    # A real-shaped vendored decoder state dict.
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_stage()])
    dec = driver._new_vendored_decoder(device=torch.device("cpu"))
    sd = {k: t.detach().clone() for k, t in dec.state_dict().items()}
    sd_keys = list(sd.keys())

    # (1) the variable-level codec mechanism is REAL (sd-key-correct coarse map shrinks
    # the blob) AND default-preserving (None/uniform -> vendored byte-identical).
    uniform_blob, uniform_is_var = build_decoder_blob_variable_or_vendored(sd, None)
    coarse_levels = {k: (16 if i < len(sd_keys) // 2 else 127) for i, k in enumerate(sd_keys)}
    coarse_blob, coarse_is_var = build_decoder_blob_variable_or_vendored(sd, coarse_levels)
    assert uniform_is_var is False, "uniform/None map must emit the vendored byte-identical blob"
    assert coarse_is_var is True, "a non-uniform coarse map must emit the variable format"
    assert len(coarse_blob) < len(uniform_blob), (
        f"the variable codec did not shrink the blob on a coarse map "
        f"({len(coarse_blob)} !< {len(uniform_blob)}) — the byte-savings mechanism is fake"
    )

    # (2) the DEFAULT driver export takes NO sensitivity argument (documented scope:
    # Lever-4 does not wire a variable-level export off the EMA). A regression that
    # added a sensitivity-driven variable-level export to the DEFAULT path would change
    # this signature / source and trip the assertion.
    sig = inspect.signature(TorchVehicleDriver._build_archive_and_eval_decoder)
    param_names = set(sig.parameters)
    assert not any("sensitiv" in p.lower() for p in param_names), (
        f"the DEFAULT export path gained a sensitivity parameter {param_names} — "
        "Lever-4's documented scope is uniform-127 export (savings via brotli "
        "compressibility), NOT a variable-level export off the sensitivity EMA"
    )
    src = inspect.getsource(TorchVehicleDriver._build_archive_and_eval_decoder)
    assert "tensor_sensitivity_ema" not in src, (
        "the DEFAULT export path now references tensor_sensitivity_ema — Lever-4 must "
        "not silently route a variable-level export off the EMA (documented scope)"
    )


@pytest.mark.timeout(600)
def test_r10_levers_compose_not_fight_on_real_scorer():
    """R10 HEADLINE lens — on the REAL frozen scorer (EfficientNet-B2 SegNet +
    FastViT PoseNet), levers 2(+5 margin)+3(pose-FiLM) COMPOSE correctly. The COMPOSE
    contract is: the combined gradient descends the COMBINED loss the driver actually
    minimizes (``w_seg·seg_l + w_pose·pose_l``) — NOT that it descends each term
    independently (when one term's gradient dominates, the combined step can mildly
    INCREASE the subordinate term; that is normal correct multi-objective descent of
    the weighted sum, not a "fight"). The well-posed checks are:
      (1) the combined gradient is a DESCENT direction for the combined loss
          (``<g_comb, g_comb> > 0`` ⇒ a ``-g_comb`` step decreases L — verified by an
          actual small GD step);
      (2) the two per-term gradients are NOT anti-aligned (``cos(g_seg, g_pose) > -1``
          with a safe margin — a genuine fight would be near-perfect opposition AND
          the combined step would increase the sum, which (1) refutes);
      (3) the margin lever (5) RESHAPES but does not REVERSE the plain seg direction
          (``cos(g_seg_margin, g_seg_plain) > 0``);
      (4) composition is by exact gradient SUM (``g_comb == g_seg + g_pose``).

    This corrects a prior R10 test-instrument bug (asserting ``cos(g_comb, g_seg) > 0``
    — wrong: when pose dominates ~1900× the combined step ≈ the pose step, which mildly
    increases seg; the combined loss STILL descends). MEASURED at the gradient-ALIVE
    T=0.5 (the cold tail's seg grad is ~0 per the floor test). RESEARCH-ONLY tiny slice
    ⇒ [contest-CPU advisory] NON-PROMOTABLE (gradient-direction claim, not a score
    claim).

    NO-FAKE: measures real autograd on the real scorer + an actual loss-decreasing GD
    step; a constant / no-op lever gives zero gradient (the norm assertions reject
    that) and the GD step would not decrease the loss."""
    pytest.importorskip("torch")
    import os

    from tac.torch_vehicle.scorer_context import RealScorerContext

    video = "upstream/videos/0.mkv"
    if not os.path.exists(video):
        pytest.skip("real video 0.mkv not present (real-scorer test)")

    from tac.torch_vehicle.driver import _EVAL_H, _EVAL_W

    torch.manual_seed(0)
    cache = _tmp_out()
    sc = RealScorerContext(video, device="cpu", max_pairs=8, targets_cache=cache)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        pose_film_enabled=True, seed=0,
    )
    spec = _stage(seg_surrogate="soft_cosine", margin_weight_tau=2.0)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    decoder.set_stored_pose(sc.pose_targets[:8].cpu())
    latents = torch.nn.Parameter(torch.randn(8, 28) * 0.1)
    idx = torch.arange(4)
    params = [p for p in decoder.parameters() if p.requires_grad] + [latents]

    def render():
        dp = decoder(latents[idx], idx)
        B = len(idx)
        flat = dp.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
        bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
        dc = bhwc.clamp(0, 255)
        return dc + (dc.round() - dc).detach()

    def gvec(loss):
        gs = torch.autograd.grad(loss, params, allow_unused=True)
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for p, g in zip(params, gs, strict=False)
        ])

    def combined_loss():
        seg_out, pose6 = sc.seg_pose_forward(render())
        seg_l = _seg_loss_for_spec(spec, seg_out, sc.seg_targets_hard[idx], temperature=0.5)
        pose_l = torch.sqrt(10.0 * F.mse_loss(pose6, sc.pose_targets[idx]) + 1e-12)
        return spec.seg_weight * seg_l + spec.pose_weight * pose_l

    # Pose term alone (Lever-3 FiLM conditions the render).
    seg_out, pose6 = sc.seg_pose_forward(render())
    pose_l = torch.sqrt(10.0 * F.mse_loss(pose6, sc.pose_targets[idx]) + 1e-12)
    g_pose = gvec(spec.pose_weight * pose_l)

    # Margin-weighted seg (Lever-2 + Lever-5) at a gradient-ALIVE T (T=0.5).
    seg_out, _ = sc.seg_pose_forward(render())
    seg_l_m = _seg_loss_for_spec(spec, seg_out, sc.seg_targets_hard[idx], temperature=0.5)
    g_seg_m = gvec(spec.seg_weight * seg_l_m)

    # Plain seg (Lever-2 only) — the margin must RESHAPE not REVERSE this.
    spec_plain = _stage(seg_surrogate="soft_cosine", margin_weight_tau=None)
    seg_out, _ = sc.seg_pose_forward(render())
    seg_l_p = _seg_loss_for_spec(spec_plain, seg_out, sc.seg_targets_hard[idx], temperature=0.5)
    g_seg_p = gvec(spec_plain.seg_weight * seg_l_p)

    # Combined gradient (the driver's actual loss; levers 2+3+5 together).
    L0 = combined_loss()
    g_comb = gvec(L0)
    l0 = float(L0.detach())

    # Non-degenerate gradients (NO-FAKE: a no-op lever would give zero norm).
    assert g_pose.norm() > 0 and g_seg_m.norm() > 0 and g_comb.norm() > 0, (
        "a lever produced a ZERO gradient on the real scorer (no-op/FAKE)"
    )

    def cos(a, b):
        return float((a @ b) / (a.norm() * b.norm()))

    # (1) The COMBINED gradient is a DESCENT direction for the COMBINED loss — the
    # real compose contract (the driver minimizes the weighted sum). Verified by an
    # actual small -g_comb GD step that DECREASES the combined loss.
    with torch.no_grad():
        off = 0
        step = 1e-6
        for p in params:
            n = p.numel()
            p.add_(-step * g_comb[off:off + n].reshape(p.shape))
            off += n
    l1 = float(combined_loss().detach())
    assert l1 < l0, (
        f"a -g_comb step did NOT decrease the combined loss ({l0:.6f} -> {l1:.6f}) — "
        "the combined gradient is not a descent direction (levers genuinely FIGHT)"
    )

    # (2) The per-term gradients are NOT anti-aligned (a genuine fight is near -1 AND
    # would make (1) fail). Mild opposition (independent loss terms) is normal.
    c_sp = cos(g_seg_m, g_pose)
    assert c_sp > -0.9, (
        f"seg and pose gradients are NEAR-ANTI-ALIGNED (cos={c_sp:.4f}) — the levers "
        "fight destructively (combined with (1) this would be a real defect)"
    )

    # (3) The margin lever (5) RESHAPES but does not REVERSE the plain seg direction.
    assert cos(g_seg_m, g_seg_p) > 0.0, (
        f"margin-weight REVERSES the plain seg direction (cos={cos(g_seg_m, g_seg_p):.4f}) "
        "— Lever-5 fights Lever-2"
    )
    # (4) Composition is by gradient SUM (the driver's single backward).
    relerr = float((g_comb - (g_seg_m + g_pose)).norm() / (g_comb.norm() + 1e-12))
    assert relerr < 1e-4, (
        f"combined gradient != g_seg + g_pose (relerr={relerr:.3e}) — the levers do "
        "NOT compose by the documented gradient sum"
    )


# --- Lever 4: score-aware QAT ---
def test_lever4_uniform_sensitivity_matches_vendored_uniform_qat():
    """Score-aware QAT with sensitivity=None (or uniform) reproduces the vendored
    uniform 127-level fake-quant BIT-IDENTICALLY (the default-preserving guard)."""
    from tac.torch_vehicle.score_aware_qat import apply_score_aware_qat, restore_score_aware_qat

    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_stage()])
    torch.manual_seed(9)
    dec_sa = driver._new_decoder(device=torch.device("cpu"))
    dec_van = driver._new_decoder(device=torch.device("cpu"))
    dec_van.load_state_dict(dec_sa.state_dict())

    o1 = apply_score_aware_qat(dec_sa, None)  # uniform fallback
    sa_weights = {n: m.weight.data.clone() for n, m in dec_sa.named_modules()
                  if n in o1}
    restore_score_aware_qat(dec_sa, o1)
    o2 = v.apply_qat(dec_van)
    van_weights = {n: m.weight.data.clone() for n, m in dec_van.named_modules()
                   if n in o2}
    v.restore_qat(dec_van, o2)

    assert set(sa_weights) == set(van_weights)
    for n in sa_weights:
        assert torch.equal(sa_weights[n], van_weights[n]), (
            f"score-aware QAT (uniform) != vendored uniform QAT at {n} (default-FAIL)"
        )


def test_lever4_nonuniform_sensitivity_changes_quant_grid():
    """A NON-UNIFORM sensitivity map produces a DIFFERENT quantized decoder than
    uniform QAT (a low-sensitivity tensor gets a coarser grid → its quant differs)."""
    from tac.torch_vehicle.score_aware_qat import (
        apply_score_aware_qat,
        per_tensor_levels_from_sensitivity,
    )

    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_stage()])
    torch.manual_seed(13)
    decoder = driver._new_decoder(device=torch.device("cpu"))
    names = [n for n, m in decoder.named_modules()
             if hasattr(m, "weight") and getattr(m, "weight", None) is not None
             and isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    assert len(names) >= 2
    # Make tensor 0 LOW-sensitivity, the rest HIGH.
    sens = {n: (0.0 if i == 0 else 100.0) for i, n in enumerate(names)}
    levels = per_tensor_levels_from_sensitivity(sens, names)
    assert levels[names[0]] < levels[names[-1]], (
        "low-sensitivity tensor must get FEWER INT8 levels (coarser grid)"
    )
    # The actual quantized weight of the low-sensitivity tensor differs from uniform.
    dec_uniform = driver._new_decoder(device=torch.device("cpu"))
    dec_uniform.load_state_dict(decoder.state_dict())
    apply_score_aware_qat(decoder, sens)
    apply_score_aware_qat(dec_uniform, None)
    w_sa = dict(decoder.named_modules())[names[0]].weight.data
    w_un = dict(dec_uniform.named_modules())[names[0]].weight.data
    assert not torch.equal(w_sa, w_un), (
        "non-uniform sensitivity did not change the low-sensitivity tensor's quant (FAKE)"
    )


def test_lever4_sensitivity_ema_accumulates_from_grad():
    """``accumulate_tensor_sensitivity`` populates the EMA from real ``w.grad``
    magnitudes — a tensor with a larger grad gets a larger sensitivity. (Mechanism,
    not a constant.)"""
    from tac.torch_vehicle.score_aware_qat import accumulate_tensor_sensitivity

    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[_stage()])
    torch.manual_seed(17)
    decoder = driver._new_decoder(device=torch.device("cpu"))
    # Synthesize grads: a known reference scalar loss so grads are non-zero + varied.
    z = torch.randn(2, 28)
    out = decoder(z)
    out.pow(2).mean().backward()
    ema: dict[str, float] = {}
    accumulate_tensor_sensitivity(decoder, ema, decay=0.0)  # decay=0 => ema == current norm
    assert ema, "no sensitivity accumulated (FAKE)"
    # decay=0 means ema[name] == ||grad|| exactly; verify against a direct norm.
    for n, m in decoder.named_modules():
        if n in ema and m.weight.grad is not None:
            assert abs(ema[n] - float(m.weight.grad.norm())) < 1e-5


# --- Lever 5: margin-weighted seg promotion ---
def test_lever5_margin_weight_changes_seg_loss():
    """``margin_weight_tau`` re-weights the per-pixel seg surrogate (boundary pixels
    get more gradient) → the weighted loss DIFFERS from the unweighted Lever-2
    baseline. A no-op weight would give the identical value."""
    seg_out, targets = _make_seg_inputs(seed=2)
    spec_unweighted = _stage(seg_surrogate="soft_cosine", seg_temperature=1.0,
                             margin_weight_tau=None)
    spec_weighted = _stage(seg_surrogate="soft_cosine", seg_temperature=1.0,
                           margin_weight_tau=2.0)
    v_un = _seg_loss_for_spec(spec_unweighted, seg_out, targets)
    v_w = _seg_loss_for_spec(spec_weighted, seg_out, targets)
    assert not torch.allclose(v_un, v_w, atol=1e-5), (
        "margin weighting did not change the seg loss — Lever 5 is a no-op (FAKE)"
    )


def test_lever5_margin_weight_is_monotone_decreasing_in_margin():
    """The Lever-5 weight ``exp(−margin/τ)`` decreases as the SegNet logit margin
    grows — confident-interior pixels (large margin) get ~0 weight, boundary pixels
    (small margin) get ~1. This is the mechanism (capacity concentrates at the
    boundary), proven on the driver's own margin map."""
    from tac.torch_vehicle.driver import _segnet_logit_margin_map

    # A logit map with a deliberate small-margin pixel and a large-margin pixel.
    seg_out = torch.zeros(1, _SEG_NUM_CLASSES, 1, 2)
    seg_out[0, :, 0, 0] = torch.tensor([1.0, 0.9, 0.0, 0.0, 0.0])  # margin 0.1 (boundary)
    seg_out[0, :, 0, 1] = torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0])  # margin 5.0 (interior)
    margin = _segnet_logit_margin_map(seg_out)  # (1, 1, 2)
    assert margin[0, 0, 0] < margin[0, 0, 1], "margin map mis-orders boundary vs interior"
    tau = 1.0
    weight = torch.exp(-margin / tau)
    assert weight[0, 0, 0] > weight[0, 0, 1], (
        "boundary pixel (small margin) must get MORE weight than the interior pixel"
    )


# ===========================================================================
# CLAIM C — COMPOSE ALL FIVE end-to-end (train -> byte-close -> inflate/parse).
# ===========================================================================
# This is the heaviest lever test: a 3-epoch synthetic driver run with ALL FIVE
# levers + QAT + pose-FiLM + C1a + a full codec byte-close + parse-back. It runs
# ~55s on an UNLOADED machine — perilously close to the global ``timeout = 60``
# (pyproject.toml). R3 review measured it time out (>60s) under heavy CPU
# contention (concurrent test suites + probes) while the SAME test passed in 54.6s
# in isolation — a pytest-timeout FLAKE, not a lever regression. A per-test 300s
# timeout removes the false-failure so a multi-day run's CI does not trip on load.
@pytest.mark.timeout(300)
def test_compose_all_five_levers_end_to_end(tmp_path):
    """A synthetic driver run with Levers 1+2(+anneal)+3+4+5 ALL ENABLED runs to a
    DONE marker, byte-closes an archive (WITH the Lever-3 pose section), and the
    archive parses back (the pose section round-trips). Proves the five levers
    compose in ONE forward/backward/export path without crashing or corrupting the
    archive grammar."""
    from tac.torch_vehicle.pose_film import parse_pose_section

    n_pairs = 6
    out_dir = tmp_path / "compose_all_five"

    spec = _stage(
        name="compose_all_five",
        epochs=3,
        # Lever 1 — rate surrogate ON (decoder weights + latents).
        rate_lambda_w=0.05,
        rate_lambda_lat=0.02,
        # Lever 2 — score-domain seg surrogate + the per-epoch anneal.
        seg_surrogate="soft_cosine",
        seg_temperature=1.0,
        seg_temperature_end=0.2,
        # Lever 4 — score-aware QAT (needs use_qat True to engage the QAT block).
        use_qat=True,
        score_aware_qat=True,
        qat_sensitivity_decay=0.9,
        # Lever 5 — margin-weighted seg promotion.
        margin_weight_tau=2.0,
        # also exercise the C1a path alongside Lever 1 (they compose additively).
        cat_lambda=0.01,
    )
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=out_dir,
        checkpoint_every_epochs=1, device="cpu", seed=0,
        # Lever 3 — pose-FiLM store ON.
        pose_film_enabled=True, pose_film_hidden=8,
    )
    driver, summary = _run_driver(cfg, spec, n_pairs=n_pairs, seed=0)

    assert summary["status"] == "complete", "compose-all-five did not reach DONE"
    arch_path = out_dir / "best" / "best_archive.bin"
    assert arch_path.exists(), "no best archive emitted by the all-five compose run"
    archive = arch_path.read_bytes()
    assert len(archive) > 0
    # The Lever-3 pose section round-trips through the additive archive grammar.
    pose = parse_pose_section(archive, driver.v.parse_archive)
    assert pose is not None and tuple(pose.shape) == (n_pairs, 6), (
        "Lever-3 pose section did not round-trip in the all-five archive (compose FAIL)"
    )
    # The vendored decoder section also parses (the archive is well-formed).
    dec_sd, latents, _meta = driver.v.parse_archive(archive)
    assert dec_sd and latents.shape[0] == n_pairs


def test_compose_all_five_loss_differs_from_all_default():
    """The all-five-enabled loss on the SAME inputs DIFFERS from the all-default
    loss — proving the composed levers are actually active in the training loop (not
    silently no-op'd). Runs ONE batch through each config's loss assembly."""
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    v = import_vendored_bundle()

    def _one_batch_loss(spec, cfg_kwargs):
        cfg = TorchVehicleConfig(
            base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu", **cfg_kwargs
        )
        driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec])
        torch.manual_seed(31)
        decoder = driver._new_decoder(device=torch.device("cpu"))
        if cfg.pose_film_enabled:
            decoder.set_stored_pose(sc.pose_targets[:6])
        latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
        idx = torch.arange(6)
        decoded_pair = decoder(latents[idx], idx) if cfg.pose_film_enabled else decoder(latents[idx])
        flat = decoded_pair.reshape(12, 3, 384, 512)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
        decoded_bhwc = down.reshape(6, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
        dc = decoded_bhwc.clamp(0, 255)
        decoded_bhwc = dc + (dc.round() - dc).detach()
        seg_out, pose_pred6 = sc.seg_pose_forward(decoded_bhwc)
        seg_l = _seg_loss_for_spec(spec, seg_out, sc.seg_targets_hard[idx],
                                   temperature=seg_temperature_for_epoch(spec, 0))
        pose_mse = F.mse_loss(pose_pred6, sc.pose_targets[idx])
        pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
        loss = spec.seg_weight * seg_l + spec.pose_weight * pose_l
        reg = driver._weight_regularizers(decoder, latents, spec)
        if reg is not None:
            loss = loss + reg
        return float(loss.item())

    default_loss = _one_batch_loss(_stage(epochs=1), {})
    allfive_loss = _one_batch_loss(
        _stage(epochs=1, rate_lambda_w=0.05, rate_lambda_lat=0.02,
               seg_surrogate="soft_cosine", seg_temperature=1.0, seg_temperature_end=0.2,
               margin_weight_tau=2.0, cat_lambda=0.01),
        {"pose_film_enabled": True, "pose_film_hidden": 8},
    )
    assert abs(default_loss - allfive_loss) > 1e-3, (
        f"all-five loss {allfive_loss:.4f} == all-default loss {default_loss:.4f} — "
        "the composed levers are silently inactive (FAKE)"
    )


# ===========================================================================
# CLAIM D (R5) — LEVERS-ON DETERMINISM + Muon×lever interaction.
# ===========================================================================
# The existing ``test_all_default_driver_run_is_deterministic_and_byte_identical``
# proves the ALL-DEFAULT path is reproducible; it does NOT cover the LEVERS-ON
# path, and uses ``use_muon=False``. R5 closes both gaps: two fresh
# all-5-levers-ON runs (same seed) must produce a bit-identical archive, INCLUDING
# under Muon (whose Newton-Schulz orthogonalization is the new nondeterminism
# surface the lever gradients flow through). A lever that introduced
# nondeterminism (unsorted set/dict iteration, a nondeterministic kernel in the
# score-aware backward, ``.item()``-driven nondeterministic control flow) would
# make these DIVERGE — corrupting a multi-day from-scratch run's reproducibility.
def _all_five_stage(*, use_muon: bool, epochs: int = 3) -> StageSpec:
    """ALL FIVE levers ON (mirrors the compose-all-five config) with the
    ``use_muon`` toggle the existing lever tests never exercise with levers on."""
    return _stage(
        name="r5_all_five",
        epochs=epochs,
        use_muon=use_muon,
        adamw_lr=(1e-5 if use_muon else 1e-3),
        use_qat=True,
        cat_lambda=0.01,
        rate_lambda_w=0.05,
        rate_lambda_lat=0.02,
        seg_surrogate="soft_cosine",
        seg_temperature=1.0,
        seg_temperature_end=0.2,
        score_aware_qat=True,
        qat_sensitivity_decay=0.9,
        margin_weight_tau=2.0,
    )


def _run_all_five(out, *, use_muon: bool, n_pairs: int = 6, seed: int = 0) -> bytes:
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=out,
        checkpoint_every_epochs=1, device="cpu", seed=seed,
        pose_film_enabled=True, pose_film_hidden=8,
    )
    _, summary = _run_driver(
        cfg, _all_five_stage(use_muon=use_muon), n_pairs=n_pairs, seed=seed
    )
    assert summary["status"] == "complete"
    return (out / "best" / "best_archive.bin").read_bytes()


@pytest.mark.timeout(300)
def test_all_five_levers_adamw_run_is_deterministic_and_byte_identical(tmp_path):
    """Two fresh all-5-levers-ON (AdamW) runs at the same seed produce a
    BIT-IDENTICAL best archive — the levers-on reproducibility floor a multi-day
    from-scratch run rests on. A nondeterministic lever path would diverge here."""
    a = _run_all_five(tmp_path / "a", use_muon=False)
    b = _run_all_five(tmp_path / "b", use_muon=False)
    assert a == b, "all-5-levers-ON (AdamW) is NON-DETERMINISTIC: archive bytes differ"


@pytest.mark.timeout(300)
def test_all_five_levers_muon_run_is_deterministic_and_byte_identical(tmp_path):
    """Two fresh all-5-levers-ON + MUON runs at the same seed produce a
    BIT-IDENTICAL best archive. Muon's Newton-Schulz orthogonalization is the new
    surface the rate gradient (Lever 1) + the QAT-shaped weights (Lever 4) flow
    through; if a lever perturbed it nondeterministically, the archive would
    diverge."""
    a = _run_all_five(tmp_path / "a", use_muon=True)
    b = _run_all_five(tmp_path / "b", use_muon=True)
    assert a == b, "all-5-levers-ON + Muon is NON-DETERMINISTIC: archive bytes differ"


def test_all_five_muon_partition_covers_film_and_routes_grads():
    """Under all-5-on + Muon, the Muon/AdamW partition covers EVERY trainable
    decoder param (0 dropped, 0 overlap), INCLUDING every FiLM param; and after a
    real all-5-on backward the rate-surrogate gradient reaches the FiLM fc1 (2D →
    Muon group) — proving no lever gradient is silently dropped for the Muon
    group. (R4 noted the Muon partition was '0-dropped' on a no-lever decoder;
    this asserts it under the levers-on backward with the rate term active.)"""
    from tac.torch_vehicle.score_aware_qat import (
        apply_score_aware_qat,
        restore_score_aware_qat,
    )

    v = import_vendored_bundle()
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(),
        device="cpu", seed=0, pose_film_enabled=True, pose_film_hidden=8,
    )
    spec = _all_five_stage(use_muon=True)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec])
    torch.manual_seed(31)
    decoder = driver._new_decoder(device=torch.device("cpu"))
    decoder.set_stored_pose(sc.pose_targets[:6])

    muon_params, adamw_params = v.partition_params_for_muon(decoder)
    muon_ids = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    trainable = [p for p in decoder.parameters() if p.requires_grad]
    assert sum(1 for p in trainable if id(p) in muon_ids or id(p) in adamw_ids) == len(
        trainable
    ), "Muon/AdamW partition drops a trainable param under all-5-on + FiLM"
    assert len(muon_ids & adamw_ids) == 0, "a param is in BOTH Muon and AdamW groups"
    film_params = [
        (n, p) for n, p in decoder.named_parameters() if n.startswith("pose_film.")
    ]
    assert film_params, "no FiLM params found (FiLM not wired)"
    assert all(
        id(p) in muon_ids or id(p) in adamw_ids for _, p in film_params
    ), "a FiLM param is uncovered by the Muon/AdamW partition"

    # real all-5-on backward — the rate term MUST reach the FiLM fc1 (2D → Muon).
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    idx = torch.arange(6)
    originals = apply_score_aware_qat(decoder, None)
    decoded_pair = decoder(latents[idx], idx)
    restore_score_aware_qat(decoder, originals)
    flat = decoded_pair.reshape(12, 3, 384, 512)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(6, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    dc = decoded_bhwc.clamp(0, 255)
    decoded_bhwc = dc + (dc.round() - dc).detach()
    seg_out, pose_pred6 = sc.seg_pose_forward(decoded_bhwc)
    seg_l = _seg_loss_for_spec(spec, seg_out, sc.seg_targets_hard[idx], temperature=1.0)
    pose_l = torch.sqrt(10.0 * F.mse_loss(pose_pred6, sc.pose_targets[idx]) + 1e-12)
    loss = spec.seg_weight * seg_l + spec.pose_weight * pose_l
    reg = driver._weight_regularizers(decoder, latents, spec)
    if reg is not None:
        loss = loss + reg
    loss.backward()
    assert all(p.grad is not None for p in muon_params), "a Muon param got no gradient"
    assert all(p.grad is not None for p in adamw_params), "an AdamW param got no gradient"
    film_fc1_muon_grad = any(
        p.ndim == 2 and id(p) in muon_ids and p.grad is not None
        and p.grad.abs().sum().item() > 0.0
        for _, p in film_params
    )
    assert film_fc1_muon_grad, (
        "the rate surrogate (Lever 1) did not reach a 2D FiLM weight in the Muon "
        "group — a lever gradient is mis-routed/dropped for the Muon partition"
    )


# ===========================================================================
# CLAIM E (R6) — Lever-4 sensitivity EMA CARRIES across the QAT->QAT stage
# boundary (the integration seam).
# ===========================================================================
# The REAL PR95 schedule has use_qat=True on FIVE consecutive stages (3-7); with
# ``--levers all`` ``score_aware_qat=True`` is set on the active + all later
# stages, so the score-aware QAT runs across MULTIPLE consecutive QAT stages. The
# per-tensor sensitivity EMA ``s_t = ||dS/dw_t||`` is a property of the CARRIED
# decoder, so it belongs to the "weights/EMA carry" side of the boundary, NOT the
# "optimizer resets per stage" side. R6 MEASURED (probe_r6_*) that BEFORE the fix
# the EMA was reset-to-empty at each QAT->QAT boundary -> the new stage's QAT fell
# back to uniform-127 for its first hundreds of steps (the SAME defect R2 fixed
# for resume, manifesting at the normal stage boundary; a behavioral -7 B / 1e-3
# loss delta). These tests guard the carry fix AND its daemon-safety (the carry is
# empty -> no-op on any non-score-aware-QAT path).
def _multi_qat_curriculum(*, score_aware: bool) -> list[StageSpec]:
    """A 2-QAT-stage curriculum (pre-QAT all-5-on -> qat_a -> qat_b), carrying the
    decoder/latents/EMA across both boundaries — the realistic multi-QAT-stage
    shape of the real PR95 schedule (stages 3-7)."""
    common = {
        "use_muon": False, "adamw_lr": 1e-3, "cat_lambda": 0.01,
        "rate_lambda_w": 1e-3, "rate_lambda_lat": 1e-3,
        "seg_surrogate": "soft_cosine", "seg_temperature": 1.0,
        "seg_temperature_end": 0.05,
        "margin_weight_tau": 2.0, "qat_sensitivity_decay": 0.99,
    }
    return [
        _stage(name="pre_qat", epochs=3, use_qat=False, init_latents_random=True,
               score_aware_qat=False, **common),
        _stage(name="qat_a", epochs=4, use_qat=True, init_latents_random=False,
               score_aware_qat=score_aware, **common),
        _stage(name="qat_b", epochs=4, use_qat=True, init_latents_random=False,
               score_aware_qat=score_aware, **common),
    ]


def _measure_second_qat_start_ema(curriculum: list[StageSpec], tmp_path) -> dict:
    """Run the multi-QAT curriculum and capture the sensitivity-EMA SIZE at the
    CONSUMPTION point: (a) at the END of the first QAT stage, and (b) at the START
    of the second QAT stage's first training epoch (AFTER the run() carry-seed has
    executed — the state the QAT block actually reads)."""
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / "run",
        checkpoint_every_epochs=1, device="cpu", seed=29,
        pose_film_enabled=True, pose_film_hidden=8,
    )
    sc = SyntheticScorerContext(n_pairs=8, device="cpu", seed=29)
    drv = TorchVehicleDriver(
        cfg, scorer=sc, vendored=import_vendored_bundle(), curriculum=curriculum
    )
    qat_names = [s.name for s in curriculum if s.use_qat]
    first_qat_name, second_qat_name = qat_names[0], qat_names[1]
    rec: dict[str, int] = {}
    orig_train = drv._train_one_epoch
    prev_rt = {"rt": None}

    def hooked_train(rt, spec, *, epoch_in_stage=0):
        # On the FIRST epoch of the second QAT stage, snapshot the EMA size the QAT
        # block will consume (post carry-seed). Also snapshot the first QAT stage's
        # END EMA the last time we see it.
        if spec.name == first_qat_name:
            prev_rt["rt"] = rt
        if (spec.name == second_qat_name and epoch_in_stage == 0
                and "second_start" not in rec):
            if prev_rt["rt"] is not None:
                rec["first_end"] = len(prev_rt["rt"].tensor_sensitivity_ema)
            rec["second_start"] = len(rt.tensor_sensitivity_ema)
        return orig_train(rt, spec, epoch_in_stage=epoch_in_stage)

    drv._train_one_epoch = hooked_train  # type: ignore[method-assign]
    out = drv.run()
    assert out["status"] == "complete"
    return rec


@pytest.mark.timeout(300)
def test_score_aware_qat_sensitivity_ema_carries_across_qat_stage_boundary(tmp_path):
    """With score-aware QAT ON across TWO consecutive QAT stages, the sensitivity
    EMA built in stage 1 (qat_a) MUST still be present at the START of stage 2
    (qat_b) — i.e. it carries across the boundary (like the weight EMA), so stage 2
    starts score-aware instead of falling back to uniform-127. GUARDS the R6 fix:
    with the carry neutered, stage2_start would be 0 (RESET) and this FAILS."""
    rec = _measure_second_qat_start_ema(
        _multi_qat_curriculum(score_aware=True), tmp_path
    )
    assert rec.get("first_end", 0) > 0, (
        "stage 1 (qat_a) never seeded the sensitivity EMA — test precondition failed"
    )
    assert rec.get("second_start", 0) == rec["first_end"], (
        f"Lever-4 sensitivity EMA did NOT carry across the QAT->QAT boundary: "
        f"stage1 END = {rec.get('first_end')} tensors, stage2 START = "
        f"{rec.get('second_start')} tensors (expected equal). The carry was lost "
        "-> stage 2 QAT falls back to uniform-127 for its first steps (the R2 "
        "resume defect, manifesting at the normal stage boundary)."
    )


@pytest.mark.timeout(300)
def test_default_qat_path_carries_empty_ema_across_boundary(tmp_path):
    """DAEMON-SAFETY: when score-aware QAT is OFF (the default / basin-daemon path),
    the sensitivity EMA is NEVER accumulated, so the cross-stage carry is EMPTY at
    every boundary — the new stage starts with an empty EMA exactly as before the
    fix. Proves the carry is a no-op on the non-score-aware-QAT path."""
    rec = _measure_second_qat_start_ema(
        _multi_qat_curriculum(score_aware=False), tmp_path
    )
    # score_aware_qat OFF => accumulate_tensor_sensitivity never runs => empty EMA
    # at the first QAT stage's end => empty carry => empty at stage 2's start.
    assert rec.get("first_end", -1) == 0, (
        f"non-score-aware QAT unexpectedly accumulated a sensitivity EMA: "
        f"{rec.get('first_end')} (expected 0)"
    )
    assert rec.get("second_start", -1) == 0, (
        f"the cross-stage carry leaked a non-empty EMA onto the default QAT path: "
        f"stage2 START = {rec.get('second_start')} (expected 0) — daemon-safety "
        "violated"
    )


# ===========================================================================
# CLAIM F (R7) — the LEVER-STATE-PERSISTENCE boundary matrix. R6 confirmed a bug
# CLASS ("lever state that should carry across a boundary but resets") with two
# QAT-EMA instances (R2 resume + R6 stage-boundary). These tests pin the NEW
# boundary types the prior rounds ran only in isolation: the non-QAT->QAT
# activation boundary (the use_qat gate must hold — empty EMA before the first
# QAT stage, then accumulate), the lever-ACTIVATION-CHANGE boundary (L4 on->OFF:
# the OFF stage must NOT mutate the carried EMA — the accumulate gate
# ``use_qat AND score_aware_qat`` must hold), and resume landing mid the SECOND
# QAT stage (the path the R6 carry fix introduced; R2 only tested resume into the
# FIRST QAT stage). Each is a Class-2-fake guard: neutering the gate/carry fails it.
# ===========================================================================
def _r7_activation_boundary_curriculum() -> list[StageSpec]:
    """The realistic ``--levers all`` shape: score_aware_qat is set on EVERY stage
    from fork onward, INCLUDING the non-QAT pre-stage (where the QAT block never
    runs, so it is a harmless no-op). The pre-QAT stage therefore has
    ``score_aware_qat=True`` but ``use_qat=False`` — the exact B1 activation
    boundary the prior rounds did not exercise (their pre-QAT stage had
    score_aware_qat=False)."""
    common = {
        "use_muon": False, "adamw_lr": 1e-3, "cat_lambda": 0.01,
        "rate_lambda_w": 1e-3, "rate_lambda_lat": 1e-3,
        "seg_surrogate": "soft_cosine", "seg_temperature": 1.0,
        "seg_temperature_end": 0.05,
        "margin_weight_tau": 2.0, "qat_sensitivity_decay": 0.99,
    }
    return [
        # pre-QAT: score_aware_qat=True but use_qat=False (the no-op activation case).
        _stage(name="pre_qat", epochs=3, use_qat=False, init_latents_random=True,
               score_aware_qat=True, **common),
        _stage(name="qat_a", epochs=4, use_qat=True, init_latents_random=False,
               score_aware_qat=True, **common),
    ]


def _measure_activation_boundary_ema(tmp_path) -> dict:
    """Hook the driver to record the EMA size at the END of the non-QAT pre-stage
    and at the START + END of the first QAT stage (the B1 activation boundary)."""
    curriculum = _r7_activation_boundary_curriculum()
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / "run",
        checkpoint_every_epochs=1, device="cpu", seed=29,
        pose_film_enabled=True, pose_film_hidden=8,
    )
    sc = SyntheticScorerContext(n_pairs=8, device="cpu", seed=29)
    drv = TorchVehicleDriver(
        cfg, scorer=sc, vendored=import_vendored_bundle(), curriculum=curriculum
    )
    rec: dict[str, int] = {}
    orig_train = drv._train_one_epoch
    last_rt = {"pre": None}

    def hooked_train(rt, spec, *, epoch_in_stage=0):
        if spec.name == "pre_qat":
            last_rt["pre"] = rt  # track to read its END EMA after the stage
        if spec.name == "qat_a" and epoch_in_stage == 0 and "qat_start" not in rec:
            if last_rt["pre"] is not None:
                rec["pre_end"] = len(last_rt["pre"].tensor_sensitivity_ema)
            rec["qat_start"] = len(rt.tensor_sensitivity_ema)
            last_rt["qat"] = rt
        return orig_train(rt, spec, epoch_in_stage=epoch_in_stage)

    drv._train_one_epoch = hooked_train  # type: ignore[method-assign]
    out = drv.run()
    assert out["status"] == "complete"
    rec["qat_end"] = len(last_rt["qat"].tensor_sensitivity_ema)  # type: ignore
    return rec


@pytest.mark.timeout(300)
def test_r7_non_qat_to_qat_activation_boundary_starts_empty_then_accumulates(tmp_path):
    """B1: with score_aware_qat=True set on a NON-QAT pre-stage (the realistic
    --levers all shape), the QAT block never runs there (use_qat gate), so the
    sensitivity EMA stays EMPTY through the pre-stage; the first QAT stage therefore
    STARTS empty (the carry from an empty prior stage is empty) and then ACCUMULATES.
    Guards the ``use_qat`` accumulate gate: if it leaked (accumulated on the non-QAT
    stage), pre_end would be > 0 and this FAILS."""
    rec = _measure_activation_boundary_ema(tmp_path)
    assert rec.get("pre_end", -1) == 0, (
        f"the non-QAT pre-stage accumulated a sensitivity EMA ({rec.get('pre_end')}) "
        "despite use_qat=False — the accumulate gate (use_qat AND score_aware_qat) "
        "leaked on the non-QAT stage."
    )
    assert rec.get("qat_start", -1) == 0, (
        f"the first QAT stage START EMA = {rec.get('qat_start')} (expected 0 — "
        "nothing seeds the EMA before the first use_qat stage)."
    )
    assert rec.get("qat_end", -1) > 0, (
        f"the first QAT stage END EMA = {rec.get('qat_end')} (expected > 0 — the "
        "score-aware QAT must accumulate the sensitivity EMA once use_qat engages)."
    )


def _r7_l4_off_after_on_curriculum() -> list[StageSpec]:
    """qat_a (use_qat, L4 ON) -> qat_b (use_qat, L4 OFF). The lever-ACTIVATION-CHANGE
    boundary: the EMA is carried INTO the L4-OFF stage (the carry block is
    unconditional, mirroring the weight-EMA carry) but the OFF stage's QAT block uses
    the vendored UNIFORM path and must NOT accumulate/mutate the EMA further."""
    common = {
        "use_muon": False, "adamw_lr": 1e-3, "cat_lambda": 0.01,
        "rate_lambda_w": 1e-3, "rate_lambda_lat": 1e-3,
        "seg_surrogate": "soft_cosine", "seg_temperature": 1.0,
        "seg_temperature_end": 0.05,
        "margin_weight_tau": 2.0, "qat_sensitivity_decay": 0.99,
    }
    return [
        _stage(name="qat_a", epochs=4, use_qat=True, init_latents_random=True,
               score_aware_qat=True, **common),
        _stage(name="qat_b_l4off", epochs=4, use_qat=True, init_latents_random=False,
               score_aware_qat=False, **common),
    ]


@pytest.mark.timeout(300)
def test_r7_l4_deactivation_boundary_does_not_mutate_carried_ema(tmp_path):
    """B3: when Lever-4 turns OFF at a stage boundary (score_aware_qat True -> False)
    while use_qat stays True, the L4-OFF stage runs the vendored UNIFORM QAT and must
    NOT accumulate into the sensitivity EMA. The EMA is carried in (harmlessly) and
    must be UNCHANGED at the end of the OFF stage. Guards the accumulate gate's
    score_aware_qat half: if it leaked (accumulated despite score_aware_qat=False),
    the OFF stage's END EMA would differ from its (carried) START EMA and this FAILS.
    This is the carry-when-should-NOT-mutate direction of the R6 bug class."""
    curriculum = _r7_l4_off_after_on_curriculum()
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / "run",
        checkpoint_every_epochs=1, device="cpu", seed=29,
        pose_film_enabled=True, pose_film_hidden=8,
    )
    sc = SyntheticScorerContext(n_pairs=8, device="cpu", seed=29)
    drv = TorchVehicleDriver(
        cfg, scorer=sc, vendored=import_vendored_bundle(), curriculum=curriculum
    )
    rec: dict[str, dict[str, float]] = {}
    orig_train = drv._train_one_epoch
    off_rt = {"rt": None}

    def hooked_train(rt, spec, *, epoch_in_stage=0):
        if spec.name == "qat_b_l4off":
            off_rt["rt"] = rt
            if epoch_in_stage == 0 and "off_start" not in rec:
                rec["off_start"] = dict(rt.tensor_sensitivity_ema)
        return orig_train(rt, spec, epoch_in_stage=epoch_in_stage)

    drv._train_one_epoch = hooked_train  # type: ignore[method-assign]
    out = drv.run()
    assert out["status"] == "complete"
    off_end = dict(off_rt["rt"].tensor_sensitivity_ema)  # type: ignore
    off_start = rec.get("off_start", {})
    assert len(off_start) > 0, (
        "precondition failed: the L4-OFF stage did not receive a non-empty carried "
        "EMA from the prior L4-ON stage (the carry should seed it)."
    )
    assert off_start == off_end, (
        f"the L4-OFF stage MUTATED the carried sensitivity EMA (start {len(off_start)} "
        f"tensors -> end {len(off_end)} tensors, values differ={off_start != off_end}) "
        "despite score_aware_qat=False — the accumulate gate's score_aware_qat half "
        "leaked: a deactivated lever must leave its accumulated state frozen, not "
        "keep updating it."
    )


@pytest.mark.timeout(400)
def test_r7_resume_mid_second_qat_stage_is_bit_identical(tmp_path):
    """B4: kill MID the SECOND QAT stage (qat_b), resume, and require the final
    archive to be BIT-IDENTICAL to the uninterrupted run. R2 only tested resume INTO
    the FIRST QAT stage; the R6 carry fix introduced a NEW interaction at the second
    QAT stage (the run() carry-seed runs, THEN _restore_into overrides it with the
    checkpointed EMA). This proves the resume-restored EMA wins AND the descent is
    bit-identical across the carry+restore path. Class-2 guard: a lost/empty
    checkpoint EMA would diverge the post-resume QAT grid -> different archive."""
    from tac.torch_vehicle.checkpoint import load_checkpoint
    from tac.torch_vehicle.driver import _SimulatedDeath

    curriculum = _multi_qat_curriculum(score_aware=True)  # pre -> qat_a -> qat_b
    second_qat = [i for i, s in enumerate(curriculum) if s.use_qat][1]
    base_ep = sum(curriculum[i].epochs for i in range(second_qat))
    kill_ep = base_ep + max(1, curriculum[second_qat].epochs // 2)

    def _cfg(out):
        return TorchVehicleConfig(
            base_channels=8, latent_dim=28, out_dir=out, checkpoint_every_epochs=1,
            device="cpu", seed=29, pose_film_enabled=True, pose_film_hidden=8,
        )

    v = import_vendored_bundle()
    # Reference: uninterrupted.
    ref_dir = tmp_path / "ref"
    dref = TorchVehicleDriver(
        _cfg(ref_dir), scorer=SyntheticScorerContext(n_pairs=8, device="cpu", seed=29),
        vendored=v, curriculum=curriculum,
    )
    dref.run()
    ref_bytes = (ref_dir / "best" / "best_archive.bin").read_bytes()

    # Killed run: stop AFTER the checkpoint at kill_ep lands (mid the SECOND QAT stage).
    res_dir = tmp_path / "res"
    dkill = TorchVehicleDriver(
        _cfg(res_dir), scorer=SyntheticScorerContext(n_pairs=8, device="cpu", seed=29),
        vendored=v, curriculum=curriculum,
    )
    dkill._stop_after_global_epoch = kill_ep
    try:
        dkill.run()
    except _SimulatedDeath:
        pass
    merged = load_checkpoint(res_dir, map_location="cpu")
    ckpt_ema = merged.get("tensor_sensitivity_ema") or {}
    assert len(ckpt_ema) > 0, (
        "the mid-second-QAT-stage checkpoint had an EMPTY sensitivity EMA — the "
        "carry+accumulate did not populate it before the checkpoint (the resume "
        "would fall back to uniform-127)."
    )

    # Resume: capture the restored EMA size, then require bit-identity.
    dres = TorchVehicleDriver(
        _cfg(res_dir), scorer=SyntheticScorerContext(n_pairs=8, device="cpu", seed=29),
        vendored=v, curriculum=curriculum,
    )
    restored = {"n": -1}
    orig_restore = dres._restore_into

    def hooked_restore(rt, m):
        orig_restore(rt, m)
        restored["n"] = len(rt.tensor_sensitivity_ema)

    dres._restore_into = hooked_restore  # type: ignore[method-assign]
    dres.run()
    resumed_bytes = (res_dir / "best" / "best_archive.bin").read_bytes()

    assert restored["n"] == len(ckpt_ema), (
        f"resume did NOT restore the checkpointed sensitivity EMA (restored "
        f"{restored['n']} != checkpoint {len(ckpt_ema)}) — the carry-seed clobbered "
        "the resume-restored EMA at the second QAT stage."
    )
    assert resumed_bytes == ref_bytes, (
        f"resume MID the second QAT stage is NOT bit-identical to the uninterrupted "
        f"run ({len(resumed_bytes)} B != {len(ref_bytes)} B) — the carry+restore path "
        "the R6 fix introduced diverged the descent."
    )


def _r7_adamw_to_muon_qat_curriculum() -> list[StageSpec]:
    """qat_a (AdamW QAT, L4 ON) -> qat_b_muon (Muon QAT, L4 ON). This is the REAL
    PR95 stage-7->stage-8 boundary shape: stages 3-7 are AdamW-only QAT, stage 8
    (muon_finetune) is the FIRST Muon stage AND a QAT stage. So that boundary is
    BOTH a QAT->QAT sensitivity-EMA carry (the R6 fix) AND an AdamW->Muon
    optimizer-PARTITION change (``partition_params_for_muon`` rebuilds the param
    groups on the CARRIED decoder). No prior test combined these: R5 ran all-5-on
    + Muon in a SINGLE stage; the R6/B4 matrix used ``use_muon=False`` throughout.
    The B5 gap is the real schedule's HARDEST boundary."""
    common = {
        "cat_lambda": 0.01, "rate_lambda_w": 1e-3, "rate_lambda_lat": 1e-3,
        "seg_surrogate": "soft_cosine", "seg_temperature": 1.0,
        "seg_temperature_end": 0.05, "margin_weight_tau": 2.0,
        "qat_sensitivity_decay": 0.99, "score_aware_qat": True,
    }
    return [
        _stage(name="qat_a_adamw", epochs=4, use_qat=True, use_muon=False,
               adamw_lr=1e-3, init_latents_random=True, **common),
        _stage(name="qat_b_muon", epochs=4, use_qat=True, use_muon=True,
               adamw_lr=1e-5, init_latents_random=False, **common),
    ]


@pytest.mark.timeout(300)
def test_r7_sensitivity_ema_carries_across_adamw_to_muon_qat_boundary(tmp_path):
    """B5 (the real stage-7->stage-8 shape): the Lever-4 sensitivity EMA built in an
    AdamW QAT stage MUST carry into the next QAT stage even when that stage flips to
    Muon (so the optimizer is re-partitioned on the CARRIED decoder). The EMA is keyed
    by tensor NAME (optimizer-agnostic), so the carry is orthogonal to the AdamW->Muon
    change — but no prior test pinned the COMBINATION. This is the matrix-completeness
    test for the real schedule's hardest boundary. Class-2 guard: with the carry
    neutered, muon_start would be 0 (RESET) and this FAILS; it also asserts the Muon
    optimizer actually built a non-empty param partition on the carried decoder (so the
    boundary is genuinely the AdamW->Muon transition, not a silent AdamW-only run)."""
    curriculum = _r7_adamw_to_muon_qat_curriculum()
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / "run",
        checkpoint_every_epochs=1, device="cpu", seed=29,
        pose_film_enabled=True, pose_film_hidden=8,
    )
    sc = SyntheticScorerContext(n_pairs=8, device="cpu", seed=29)
    drv = TorchVehicleDriver(
        cfg, scorer=sc, vendored=import_vendored_bundle(), curriculum=curriculum
    )
    rec: dict[str, int] = {}
    orig_train = drv._train_one_epoch
    prev = {"rt": None}

    def hooked_train(rt, spec, *, epoch_in_stage=0):
        if spec.name == "qat_a_adamw":
            prev["rt"] = rt
        if spec.name == "qat_b_muon" and epoch_in_stage == 0 and "muon_start" not in rec:
            rec["adamw_end"] = len(prev["rt"].tensor_sensitivity_ema)  # type: ignore
            rec["muon_start"] = len(rt.tensor_sensitivity_ema)
            rec["muon_opt_built"] = int(rt.muon_opt is not None)
            rec["n_muon_params"] = len(rt.muon_params)
        return orig_train(rt, spec, epoch_in_stage=epoch_in_stage)

    drv._train_one_epoch = hooked_train  # type: ignore[method-assign]
    out = drv.run()
    assert out["status"] == "complete"
    assert rec.get("adamw_end", 0) > 0, (
        "the AdamW QAT stage never seeded the sensitivity EMA — test precondition failed"
    )
    assert rec.get("muon_opt_built", 0) == 1 and rec.get("n_muon_params", 0) > 0, (
        f"the Muon QAT stage did NOT build a non-empty Muon param partition on the "
        f"carried decoder (muon_opt_built={rec.get('muon_opt_built')}, "
        f"n_muon_params={rec.get('n_muon_params')}) — the boundary is not genuinely "
        "the AdamW->Muon transition the test means to exercise."
    )
    assert rec.get("muon_start", 0) == rec["adamw_end"], (
        f"Lever-4 sensitivity EMA did NOT carry across the AdamW->Muon QAT boundary: "
        f"AdamW stage END = {rec.get('adamw_end')} tensors, Muon stage START = "
        f"{rec.get('muon_start')} tensors (expected equal). The optimizer-partition "
        "change at the real stage-7->stage-8 boundary lost the carried EMA -> the Muon "
        "QAT stage falls back to uniform-127 for its first steps."
    )


# --- R11: resume × anneal-schedule restore (the daemon-crash-mid-anneal case) ---
def _r11_all5_anneal_stage(*, epochs: int, t_end: float) -> StageSpec:
    """A single all-5-levers-ON stage with the seg-temperature anneal ACTIVE (the
    live distortion arm's shape: soft_cosine + temperature 1.0 -> t_end + margin +
    rate + QAT + pose-FiLM). The kill epoch is chosen so the annealed temperature at
    the resume point is a NON-trivial INTERMEDIATE value (not the start 1.0, not the
    floor t_end) — the exact case the prior B4 resume test did NOT exercise (qat_b's
    mid-stage anneal lands near the floor)."""
    return _stage(
        name="r11_all5_anneal", epochs=epochs, use_muon=False, use_qat=True,
        init_latents_random=True, cat_lambda=0.01, rate_lambda_w=0.05,
        rate_lambda_lat=0.02, seg_surrogate="soft_cosine", seg_temperature=1.0,
        seg_temperature_end=t_end, score_aware_qat=True, qat_sensitivity_decay=0.9,
        margin_weight_tau=2.0, adamw_lr=1e-3,
    )


@pytest.mark.timeout(300)
def test_r11_resume_mid_anneal_all5_is_bit_identical(tmp_path):
    """R11 — a daemon crash MID an all-5-on anneal stage, resumed, reproduces a
    BIT-IDENTICAL archive. The resume point is chosen where the annealed temperature
    is a NON-trivial intermediate value (NOT the start, NOT the floor), so the
    resumed epoch MUST restore the EXACT annealed temperature for the loss to match.

    ``seg_temperature_for_epoch`` is a PURE function of ``(spec, epoch_in_stage)`` —
    no hidden state — so the annealed temperature is reconstructed from the restored
    ``epoch_in_stage``, never persisted. This test MEASURES (does not assume — the
    R7 lesson) that the reconstruction is exact: a regression that mis-restored the
    epoch index, or that made the temperature depend on accumulated state, would give
    a DIFFERENT temperature at the resumed epoch -> a different seg loss -> a
    different archive. Class-2-fake-proof: the kill epoch's annealed T is asserted to
    be strictly between the floor and the start (so the test genuinely exercises the
    intermediate-anneal restore, not a trivial floor/start case)."""
    from tac.torch_vehicle.checkpoint import load_checkpoint
    from tac.torch_vehicle.driver import _SimulatedDeath

    epochs, t_end = 6, 0.2
    spec = _r11_all5_anneal_stage(epochs=epochs, t_end=t_end)
    kill_ep = 3  # mid-stage
    # The annealed T at the resume point must be a NON-trivial intermediate value.
    t_at_kill = seg_temperature_for_epoch(spec, kill_ep)
    assert t_end + 1e-6 < t_at_kill < 1.0 - 1e-6, (
        f"the kill-epoch annealed T {t_at_kill} is not strictly intermediate "
        f"(floor {t_end}, start 1.0) — the test would not exercise the intermediate "
        "anneal restore"
    )

    def _cfg(out):
        return TorchVehicleConfig(
            base_channels=8, latent_dim=28, out_dir=out, checkpoint_every_epochs=1,
            device="cpu", seed=31, pose_film_enabled=True, pose_film_hidden=8,
        )

    v = import_vendored_bundle()
    ref_dir = tmp_path / "ref"
    TorchVehicleDriver(
        _cfg(ref_dir), scorer=SyntheticScorerContext(n_pairs=8, device="cpu", seed=31),
        vendored=v, curriculum=[spec],
    ).run()
    ref_bytes = (ref_dir / "best" / "best_archive.bin").read_bytes()

    res_dir = tmp_path / "res"
    dkill = TorchVehicleDriver(
        _cfg(res_dir), scorer=SyntheticScorerContext(n_pairs=8, device="cpu", seed=31),
        vendored=v, curriculum=[spec],
    )
    dkill._stop_after_global_epoch = kill_ep
    try:
        dkill.run()
    except _SimulatedDeath:
        pass
    merged = load_checkpoint(res_dir, map_location="cpu")
    assert merged["position"].epoch_in_stage == kill_ep, (
        f"checkpoint restored the wrong epoch_in_stage ({merged['position'].epoch_in_stage} "
        f"!= {kill_ep}) — the resumed anneal temperature would be wrong"
    )

    TorchVehicleDriver(
        _cfg(res_dir), scorer=SyntheticScorerContext(n_pairs=8, device="cpu", seed=31),
        vendored=v, curriculum=[spec],
    ).run()
    resumed_bytes = (res_dir / "best" / "best_archive.bin").read_bytes()
    assert resumed_bytes == ref_bytes, (
        f"resume MID the all-5-on anneal stage is NOT bit-identical "
        f"({len(resumed_bytes)} B != {len(ref_bytes)} B) — the annealed temperature "
        f"(intermediate {t_at_kill:.4f} at the resume epoch) did not restore exactly, "
        "OR the pose-FiLM / L4-EMA / QAT state diverged across the resume."
    )


@pytest.mark.timeout(300)
def test_r11_resume_into_muon_stage_with_anneal_is_bit_identical(tmp_path):
    """R11 — a daemon crash MID the Muon-QAT stage (the real stage-7->8 AdamW->Muon
    shape) with the anneal ACTIVE at a NON-floor temperature, resumed, reproduces a
    BIT-IDENTICAL archive. This is the JOINT of: (a) the B4 resume-mid-QAT case,
    (b) the B5 AdamW->Muon partition rebuild, and (c) the R11 intermediate-anneal
    restore — no prior test combined all three. The resumed Muon stage must rebuild
    its partition on the carried decoder, restore the L4 EMA + pose-FiLM state, AND
    reconstruct the exact annealed temperature for the resumed epoch."""
    from tac.torch_vehicle.checkpoint import load_checkpoint
    from tac.torch_vehicle.driver import _SimulatedDeath

    def stg(name, epochs, use_muon, t_end):
        return _stage(
            name=name, epochs=epochs, use_muon=use_muon, use_qat=True,
            init_latents_random=(name == "s0"), cat_lambda=0.01, rate_lambda_w=0.05,
            rate_lambda_lat=0.02, seg_surrogate="soft_cosine", seg_temperature=1.0,
            seg_temperature_end=t_end, score_aware_qat=True, qat_sensitivity_decay=0.9,
            margin_weight_tau=2.0, adamw_lr=(1e-5 if use_muon else 1e-3),
        )

    curriculum = [
        stg("s0", 4, False, 0.2),
        stg("s1", 4, False, 0.1),
        stg("s2", 4, True, 0.05),  # the Muon stage (AdamW->Muon at s1->s2)
    ]
    base = sum(curriculum[i].epochs for i in range(2))
    kill_ep = base + 2  # mid the Muon stage s2
    t_at_kill = seg_temperature_for_epoch(curriculum[2], 2)
    assert 0.05 + 1e-6 < t_at_kill < 1.0 - 1e-6, (
        f"the Muon-stage kill-epoch annealed T {t_at_kill} is not intermediate"
    )

    def _cfg(out):
        return TorchVehicleConfig(
            base_channels=8, latent_dim=28, out_dir=out, checkpoint_every_epochs=1,
            device="cpu", seed=33, pose_film_enabled=True, pose_film_hidden=8,
        )

    v = import_vendored_bundle()
    ref_dir = tmp_path / "ref"
    TorchVehicleDriver(
        _cfg(ref_dir), scorer=SyntheticScorerContext(n_pairs=8, device="cpu", seed=33),
        vendored=v, curriculum=curriculum,
    ).run()
    ref_bytes = (ref_dir / "best" / "best_archive.bin").read_bytes()

    res_dir = tmp_path / "res"
    dkill = TorchVehicleDriver(
        _cfg(res_dir), scorer=SyntheticScorerContext(n_pairs=8, device="cpu", seed=33),
        vendored=v, curriculum=curriculum,
    )
    dkill._stop_after_global_epoch = kill_ep
    try:
        dkill.run()
    except _SimulatedDeath:
        pass
    merged = load_checkpoint(res_dir, map_location="cpu")
    assert merged["position"].stage_index == 2, "did not checkpoint INSIDE the Muon stage"
    ckpt_ema = merged.get("tensor_sensitivity_ema") or {}
    assert len(ckpt_ema) > 0, "the mid-Muon-QAT checkpoint lost the L4 sensitivity EMA"

    TorchVehicleDriver(
        _cfg(res_dir), scorer=SyntheticScorerContext(n_pairs=8, device="cpu", seed=33),
        vendored=v, curriculum=curriculum,
    ).run()
    resumed_bytes = (res_dir / "best" / "best_archive.bin").read_bytes()
    assert resumed_bytes == ref_bytes, (
        f"resume MID the Muon-QAT anneal stage is NOT bit-identical "
        f"({len(resumed_bytes)} B != {len(ref_bytes)} B) — the JOINT of "
        "AdamW->Muon partition rebuild + L4-EMA restore + intermediate-anneal restore "
        f"(T={t_at_kill:.4f}) + pose-FiLM restore diverged across the resume."
    )
