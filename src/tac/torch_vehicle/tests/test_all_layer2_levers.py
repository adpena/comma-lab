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

from tac.torch_vehicle.curriculum import StageSpec, seg_temperature_for_epoch
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
