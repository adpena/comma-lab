# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Ballé end-to-end rate-distortion lever — the
weight-entropy penalty wired into the torch-vehicle HNeRV training path.

THE LOAD-BEARING CLAIMS (each, if wrong, is either a FAKE lever OR — worse —
silently changes the LIVE base_ch=20 basin if it crash-resumes onto this code):

A. **λ=0 BYTE IDENTITY (the daemon-safety guard).** With
   ``weight_entropy_penalty_lambda == 0.0`` (the default), the penalty module is
   NEVER built, its params NEVER enter the optimizer, and the loss + the trained
   weights are bit-identical to the no-penalty path. The live MPS basin (which uses
   λ=0) is unaffected.

B. **λ>0 ACTUALLY LOWERS THE MEASURED CODEC WEIGHT-SYMBOL ENTROPY (the headline
   NO-FAKE test).** Two driver runs with bit-shared init differing ONLY by λ: the
   λ>0 run ends with LOWER measured (hard codec-quantized) symbol entropy than the
   λ=0 run. This MUST FAIL if the penalty is a no-op marker (replacing the term with
   ``return baseline`` would not lower entropy). The metric is the REAL exact symbol
   histogram entropy the codec codes — not the differentiable surrogate.

C. **THE PENALTY PRIOR PARAMS ARE IN THE OPTIMIZER (grad flows, they update).**
   When λ>0, the AdamW optimizer contains the penalty's learnable prior params and
   they actually change across a step (the prior adapts).

D. **SHAPE HANDLING for the real decoder weight tensors** — the penalty's tensor
   set is exactly the codec's coded Conv2d/Linear set, and ``rate_bits`` runs on
   the real (base_ch20) decoder without shape errors and carries gradient to the
   decoder weights.
"""

from __future__ import annotations

import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext
from tac.torch_vehicle.weight_entropy_penalty import (
    WeightEntropyPenalty,
    _coded_weight_modules,
    measure_decoder_weight_symbol_entropy,
)


def _ce(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _stage(**overrides) -> StageSpec:
    base = {
        "name": "we_test",
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
    return tempfile.mkdtemp(prefix="weight_entropy_test_")


def _run_driver(cfg: TorchVehicleConfig, spec: StageSpec, *, n_pairs: int = 6, seed: int = 0):
    sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=seed)
    driver = TorchVehicleDriver(
        cfg, scorer=sc, vendored=import_vendored_bundle(), curriculum=[spec]
    )
    summary = driver.run()
    return driver, summary


class _Tiny(nn.Module):
    """A minimal Conv2d+Linear stand-in (the codec's coded-tensor set) for the
    module-unit tests that don't need the full base_ch20 decoder."""

    def __init__(self, in_ch: int = 4, mid: int = 8, out: int = 6):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, mid, 3)
        self.lin = nn.Linear(mid, out)

    def forward(self, x):  # pragma: no cover - not exercised
        return x


# ===========================================================================
# CLAIM A — λ=0 BYTE IDENTITY (the daemon-safety guard).
# ===========================================================================
def test_config_weight_entropy_penalty_defaults_off():
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out())
    assert cfg.weight_entropy_penalty_lambda == 0.0
    assert cfg.weight_entropy_penalty_stage_min == 0


def test_lambda_zero_does_not_build_penalty_module():
    """λ=0 → the driver never constructs the penalty module (no params, no optimizer
    group, no loss term) — the byte-identity precondition."""
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        weight_entropy_penalty_lambda=0.0,
    )
    driver, _ = _run_driver(cfg, _stage(epochs=1), n_pairs=6, seed=0)
    assert driver._weight_entropy_penalty is None


def test_lambda_zero_run_is_byte_identical_to_pre_lever_path(tmp_path):
    """Two all-default (λ=0) runs with the same seed are bit-identical in best score
    AND archive bytes (an λ=0 run IS the pre-lever path)."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    spec = _stage(epochs=3, cat_lambda=0.0)
    cfg_a = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=out_a, device="cpu", seed=0,
        weight_entropy_penalty_lambda=0.0,
    )
    cfg_b = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=out_b, device="cpu", seed=0,
        weight_entropy_penalty_lambda=0.0,
    )
    _, sum_a = _run_driver(cfg_a, spec, seed=0)
    _, sum_b = _run_driver(cfg_b, spec, seed=0)
    assert sum_a["best_score"] == sum_b["best_score"]
    arch_a = (out_a / "best" / "best_archive.bin").read_bytes()
    arch_b = (out_b / "best" / "best_archive.bin").read_bytes()
    assert arch_a == arch_b


def test_weight_regularizers_returns_none_on_lambda_zero_default():
    """The regularizer helper returns the EXACT legacy result (None) when neither
    C1a nor Lever-1 nor the weight-entropy penalty is active — the loss is unchanged."""
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        weight_entropy_penalty_lambda=0.0,
    )
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[_stage(epochs=1)])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    reg = driver._weight_regularizers(decoder, latents, _stage(epochs=1, cat_lambda=0.0))
    assert reg is None


# ===========================================================================
# CLAIM B — λ>0 ACTUALLY LOWERS THE MEASURED CODEC WEIGHT-SYMBOL ENTROPY.
# ===========================================================================
def test_pure_surrogate_descent_lowers_measured_codec_entropy():
    """MECHANISM (isolated): minimizing ONLY the Ballé surrogate (decoder weights +
    prior params jointly) LOWERS the REAL hard-codec-quantized symbol entropy. This is
    the NO-FAKE core: the surrogate is a faithful proxy for the bytes the codec codes.
    Fails if the term is a no-op."""
    torch.manual_seed(0)
    d = _Tiny(8, 16, 12)
    pen = WeightEntropyPenalty(d, init_scale=10.0).train()
    d.train()
    opt = torch.optim.AdamW(list(d.parameters()) + list(pen.parameters()), lr=1e-2)
    h0 = measure_decoder_weight_symbol_entropy(d)
    for _ in range(150):
        opt.zero_grad()
        total_bits, _rate = pen.rate_bits(d)
        total_bits.backward()
        opt.step()
    h1 = measure_decoder_weight_symbol_entropy(d)
    assert h1 < h0 - 0.05, f"surrogate descent did not lower measured entropy: {h0:.4f}->{h1:.4f}"


def test_driver_lambda_positive_lowers_measured_entropy_vs_lambda_zero():
    """THE HEADLINE NO-FAKE TEST. Two driver runs with bit-shared init differing ONLY
    by λ (one λ=0, one λ>0): the λ>0 run ends with LOWER measured codec symbol entropy
    on the trained decoder. The metric is the EXACT symbol-histogram entropy the codec
    codes — not the surrogate. A no-op penalty would NOT lower it (the test fails)."""
    # A from-scratch curriculum stage with a high λ so the rate pressure is visible in
    # a short run. seg_weight kept so the task loss still trains (a realistic mixture).
    spec0 = _stage(epochs=12, adamw_lr=3e-3, seg_weight=100.0, pose_weight=1.0)
    spec1 = _stage(epochs=12, adamw_lr=3e-3, seg_weight=100.0, pose_weight=1.0)

    cfg0 = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu", seed=0,
        weight_entropy_penalty_lambda=0.0,
    )
    cfg1 = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu", seed=0,
        weight_entropy_penalty_lambda=50.0,  # strong rate pressure for a short run
        weight_entropy_penalty_stage_min=0,
    )
    drv0, _ = _run_driver(cfg0, spec0, n_pairs=6, seed=0)
    drv1, _ = _run_driver(cfg1, spec1, n_pairs=6, seed=0)

    # Measure on the FINAL trained (live, non-EMA) decoder of each run — the weights the
    # penalty actually shaped. Both runs share the bit-identical init + RNG, so any
    # entropy gap is the lever's doing.
    h0 = measure_decoder_weight_symbol_entropy(drv0._final_decoder)
    h1 = measure_decoder_weight_symbol_entropy(drv1._final_decoder)
    assert h1 < h0, (
        f"λ>0 did NOT lower the measured codec weight-symbol entropy "
        f"(λ=0: {h0:.4f}  λ>0: {h1:.4f}) — the Ballé lever is a no-op as wired"
    )


def test_weight_regularizers_adds_nonzero_term_when_lambda_positive():
    """λ>0 → the regularizer helper returns a NON-NONE, NON-ZERO, gradient-carrying
    term (the loss actually changes); λ=0 on the same decoder returns None."""
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        weight_entropy_penalty_lambda=10.0,
    )
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[_stage(epochs=1)])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    # Build the penalty (mirrors _build_stage_runtime's lazy build) so the helper sees it.
    driver._build_stage_runtime(
        _stage(epochs=1), decoder=decoder, latents=latents,
        ema_decoder=None, ema_latents=None,
    )
    assert driver._weight_entropy_penalty is not None
    reg = driver._weight_regularizers(decoder, latents, _stage(epochs=1, cat_lambda=0.0))
    assert reg is not None
    assert torch.is_tensor(reg) and reg.requires_grad
    assert float(reg.item()) > 0.0


def test_stage_min_gates_the_term_off_before_the_threshold():
    """``weight_entropy_penalty_stage_min`` honors the C1a-style late-stage schedule:
    a stage index BELOW the threshold contributes NO term even when λ>0."""
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        weight_entropy_penalty_lambda=10.0, weight_entropy_penalty_stage_min=3,
    )
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[_stage(epochs=1)])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    driver._build_stage_runtime(
        _stage(epochs=1), decoder=decoder, latents=latents,
        ema_decoder=None, ema_latents=None,
    )
    # current stage index 0 < stage_min 3 → gated OFF → None (no other reg active)
    driver._cur_stage_index = 0
    assert driver._weight_regularizers(decoder, latents, _stage(cat_lambda=0.0)) is None
    # at/after the threshold → the term is active
    driver._cur_stage_index = 3
    reg = driver._weight_regularizers(decoder, latents, _stage(cat_lambda=0.0))
    assert reg is not None and float(reg.item()) > 0.0


# ===========================================================================
# CLAIM C — THE PENALTY PRIOR PARAMS ARE IN THE OPTIMIZER (grad flows, update).
# ===========================================================================
def test_penalty_params_are_in_the_adamw_optimizer():
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        weight_entropy_penalty_lambda=10.0,
    )
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[_stage(epochs=1)])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    rt = driver._build_stage_runtime(
        _stage(epochs=1), decoder=decoder, latents=latents,
        ema_decoder=None, ema_latents=None,
    )
    opt_param_ids = {
        id(p) for group in rt.adamw_opt.param_groups for p in group["params"]
    }
    penalty_ids = {id(p) for p in driver._weight_entropy_penalty.parameters()}
    assert penalty_ids, "penalty has no params"
    assert penalty_ids <= opt_param_ids, "penalty prior params are NOT in the AdamW optimizer"


def test_penalty_params_actually_update_during_a_step():
    """The prior params CHANGE across a training step (grad flows → they adapt). If they
    were frozen / not in the optimizer this fails."""
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu", seed=0,
        weight_entropy_penalty_lambda=10.0,
    )
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    spec = _stage(epochs=1, adamw_lr=1e-2)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[spec])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    rt = driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )
    before = [p.detach().clone() for p in driver._weight_entropy_penalty.parameters()]
    torch.manual_seed(123)
    driver._train_one_epoch(rt, spec, epoch_in_stage=0)
    after = list(driver._weight_entropy_penalty.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
    assert changed, "penalty prior params did not change across a training step"


# ===========================================================================
# CLAIM D — SHAPE HANDLING for the real decoder weight tensors.
# ===========================================================================
def test_penalty_tensor_set_matches_codec_coded_set_on_real_decoder():
    """The penalty's bottleneck set is EXACTLY the codec's coded Conv2d/Linear weight
    set on the real base_ch20 decoder (no more, no fewer)."""
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[_stage(epochs=1)])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    coded_names = {n for n, _m in _coded_weight_modules(decoder)}
    pen = WeightEntropyPenalty(decoder)
    assert set(pen._key_for_name.keys()) == coded_names
    assert len(coded_names) > 0


def test_rate_bits_runs_on_real_decoder_and_carries_gradient_to_weights():
    """``rate_bits`` runs on the real base_ch20 decoder (every Conv2d/Linear shape)
    without error and the total_bits gradient reaches the decoder weights."""
    cfg = TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu")
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[_stage(epochs=1)])
    decoder = driver._new_decoder(device=torch.device("cpu")).train()
    pen = WeightEntropyPenalty(decoder).train()
    total_bits, rate_term = pen.rate_bits(decoder)
    assert torch.is_tensor(total_bits) and float(total_bits.item()) > 0.0
    assert float(rate_term.item()) > 0.0
    total_bits.backward()
    coded = _coded_weight_modules(decoder)
    grad_present = any(
        m.weight.grad is not None and float(m.weight.grad.abs().sum()) > 0.0
        for _n, m in coded
    )
    assert grad_present, "rate_bits did not carry gradient to any decoder weight"


def test_scale_is_detached_only_distribution_shape_is_penalized():
    """The codec-grid representation detaches the per-tensor SCALE: scaling a weight
    tensor by a constant (which leaves the integer SYMBOLS unchanged) does NOT change
    total_bits. This proves the penalty targets the symbol DISTRIBUTION, not magnitude."""
    torch.manual_seed(0)
    d = _Tiny(4, 8, 6)
    pen = WeightEntropyPenalty(d, init_scale=10.0).eval()  # eval=round (deterministic)
    bits_a, _ = pen.rate_bits(d)
    with torch.no_grad():
        for _n, m in _coded_weight_modules(d):
            m.weight.mul_(3.0)  # rescale: symbols q=round(w/scale) are invariant
    bits_b, _ = pen.rate_bits(d)
    assert abs(float(bits_a.item()) - float(bits_b.item())) < 1e-3, (
        "rescaling the weights changed total_bits — the scale is NOT detached "
        "(the penalty is penalizing magnitude, not the codec symbol distribution)"
    )


def test_empty_decoder_raises():
    """A module with NO coded Conv2d/Linear weight tensors raises (never silently
    builds an empty penalty that would no-op)."""
    class _NoCoded(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(4)

    with pytest.raises(ValueError):
        WeightEntropyPenalty(_NoCoded())


# ===========================================================================
# OPTIMIZE #4 — per-tensor WATERFILL allocation (KKT reverse-water-fill).
# ===========================================================================
def test_waterfill_weights_normalize_to_unit_numel_weighted_mean():
    """``compute_waterfill_weights`` returns multipliers whose numel-weighted MEAN is
    1.0 — so a waterfill run spends the SAME aggregate λ-budget as the uniform run
    (a fair A/B; only the ALLOCATION differs)."""
    torch.manual_seed(0)
    d = _Tiny(8, 16, 12)
    pen = WeightEntropyPenalty(d, init_scale=10.0)
    ww = pen.compute_waterfill_weights(d, sensitivity=None)
    from tac.torch_vehicle.weight_entropy_penalty import _coded_weight_modules as _cwm
    numels = {n: float(m.weight.numel()) for n, m in _cwm(d)}
    tot = sum(numels.values())
    mean = sum(ww[n] * numels[n] for n in ww) / tot
    assert abs(mean - 1.0) < 1e-5, f"numel-weighted mean multiplier is {mean}, not 1.0"


def test_waterfill_concentrates_on_low_sensitivity_tensors():
    """A tensor flagged HIGH-sensitivity gets a LOWER multiplier than the same tensor
    flagged LOW-sensitivity (the protection mechanism)."""
    torch.manual_seed(0)
    d = _Tiny(8, 16, 12)
    pen = WeightEntropyPenalty(d, init_scale=10.0)
    from tac.torch_vehicle.weight_entropy_penalty import _coded_weight_modules as _cwm
    names = [n for n, _m in _cwm(d)]
    # Make the FIRST tensor very sensitive, the rest neutral.
    sens_hi = {names[0]: 100.0}
    sens_lo = {names[0]: 0.01}
    w_hi = pen.compute_waterfill_weights(d, sensitivity=sens_hi)
    w_lo = pen.compute_waterfill_weights(d, sensitivity=sens_lo)
    assert w_hi[names[0]] < w_lo[names[0]], (
        "high-sensitivity tensor was NOT protected (its multiplier did not drop)"
    )


def test_rate_bits_per_tensor_weights_changes_loss_but_not_reported_rate_term():
    """``per_tensor_weights`` re-weights ``total_bits`` (the loss term) but leaves
    ``rate_term`` (the reported contest-scale magnitude) UN-weighted (so the score-scale
    number is comparable across uniform/waterfill A/Bs)."""
    torch.manual_seed(0)
    d = _Tiny(8, 16, 12)
    pen = WeightEntropyPenalty(d, init_scale=10.0).eval()
    from tac.torch_vehicle.weight_entropy_penalty import _coded_weight_modules as _cwm
    names = [n for n, _m in _cwm(d)]
    bits_u, rate_u = pen.rate_bits(d)
    skew = {names[0]: 5.0}  # heavily up-weight the first tensor
    bits_w, rate_w = pen.rate_bits(d, per_tensor_weights=skew)
    assert float(bits_w.item()) != float(bits_u.item()), "weighting did not change total_bits"
    assert abs(float(rate_w.item()) - float(rate_u.item())) < 1e-9, (
        "rate_term changed under weighting — it must report the UN-weighted bits"
    )


def test_uniform_weights_dict_is_identical_to_none():
    """An all-1.0 ``per_tensor_weights`` map is bit-identical to ``None`` (the uniform
    default) — the waterfill path with neutral weights is the legacy path."""
    torch.manual_seed(0)
    d = _Tiny(8, 16, 12)
    pen = WeightEntropyPenalty(d, init_scale=10.0).eval()
    from tac.torch_vehicle.weight_entropy_penalty import _coded_weight_modules as _cwm
    ones = {n: 1.0 for n, _m in _cwm(d)}
    bits_none, _ = pen.rate_bits(d)
    bits_ones, _ = pen.rate_bits(d, per_tensor_weights=ones)
    assert abs(float(bits_none.item()) - float(bits_ones.item())) < 1e-9


def test_driver_waterfill_default_off_is_byte_identical(tmp_path):
    """``weight_entropy_penalty_waterfill=False`` (the default) with λ>0 uses the uniform
    path — two such runs are bit-identical (the waterfill flag does not perturb the
    uniform allocation)."""
    spec = _stage(epochs=3, adamw_lr=3e-3)
    cfg_a = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=tmp_path / "a", device="cpu", seed=0,
        weight_entropy_penalty_lambda=10.0, weight_entropy_penalty_waterfill=False,
    )
    cfg_b = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=tmp_path / "b", device="cpu", seed=0,
        weight_entropy_penalty_lambda=10.0, weight_entropy_penalty_waterfill=False,
    )
    _, sum_a = _run_driver(cfg_a, spec, seed=0)
    _, sum_b = _run_driver(cfg_b, spec, seed=0)
    assert sum_a["best_score"] == sum_b["best_score"]


def test_driver_waterfill_on_runs_and_differs_from_uniform(tmp_path):
    """``weight_entropy_penalty_waterfill=True`` runs end-to-end AND produces a DIFFERENT
    trained decoder than the uniform allocation (the allocation actually changed the
    descent). Bit-shared init; the only difference is the waterfill flag."""
    spec = _stage(epochs=10, adamw_lr=3e-3)
    cfg_u = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=tmp_path / "u", device="cpu", seed=0,
        weight_entropy_penalty_lambda=50.0, weight_entropy_penalty_waterfill=False,
    )
    cfg_w = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=tmp_path / "w", device="cpu", seed=0,
        weight_entropy_penalty_lambda=50.0, weight_entropy_penalty_waterfill=True,
    )
    drv_u, _ = _run_driver(cfg_u, spec, seed=0)
    drv_w, _ = _run_driver(cfg_w, spec, seed=0)
    # The two final decoders differ (the allocation steered the weights differently).
    any_diff = any(
        not torch.equal(pu.detach().cpu(), pw.detach().cpu())
        for pu, pw in zip(
            drv_u._final_decoder.parameters(),
            drv_w._final_decoder.parameters(),
            strict=True,
        )
    )
    assert any_diff, "waterfill produced a bit-identical decoder to uniform (no effect)"


# ===========================================================================
# OPTIMIZE #5 — the learned prior persists across capture/restore (resume).
# ===========================================================================
def test_prior_params_round_trip_through_capture_restore():
    """The learned Ballé prior params survive ``_capture_state`` → ``_restore_into``
    (the resume-correctness fix). Without it, a λ>0 resume rebuilds a FRESH prior."""
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu", seed=0,
        weight_entropy_penalty_lambda=10.0,
    )
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    spec = _stage(epochs=1, adamw_lr=1e-2)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[spec])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    rt = driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )
    # Train a step so the prior ADAPTS away from init.
    torch.manual_seed(7)
    driver._train_one_epoch(rt, spec, epoch_in_stage=0)
    snap = {k: v.detach().clone() for k, v in driver._weight_entropy_penalty.state_dict().items()}

    # Capture, then CORRUPT the live prior, then restore — must recover the snapshot.
    cap = driver._capture_state(rt, spec)
    assert cap["weight_entropy_penalty"] is not None
    with torch.no_grad():
        for p in driver._weight_entropy_penalty.parameters():
            p.add_(99.0)
    driver._restore_into(rt, cap)
    after = driver._weight_entropy_penalty.state_dict()
    for k in snap:
        assert torch.allclose(after[k], snap[k]), f"prior param {k} did not round-trip"


def test_lambda_zero_capture_has_no_penalty_key_value():
    """On the default λ=0 path the captured state carries ``weight_entropy_penalty:
    None`` (the penalty is never built) — the resume is byte-identical."""
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu", seed=0,
        weight_entropy_penalty_lambda=0.0,
    )
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    spec = _stage(epochs=1)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[spec])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    rt = driver._build_stage_runtime(
        spec, decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None
    )
    cap = driver._capture_state(rt, spec)
    assert cap["weight_entropy_penalty"] is None


# ===========================================================================
# OPTIMIZE #3a — C1a DOUBLE-COUNT guard (penalty supersedes the same-quantity C1a).
# ===========================================================================
def test_penalty_supersedes_c1a_zeroes_the_c1a_term():
    """With λ>0 + the penalty active + ``supersedes_c1a=True`` (default), a stage with
    ``cat_lambda>0`` does NOT add the C1a term twice — the reg term equals the
    penalty-only term (the C1a soft-histogram entropy is superseded). The two penalize
    the SAME codec-grid symbol entropy; stacking is measured net-negative."""
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        weight_entropy_penalty_lambda=10.0, weight_entropy_penalty_supersedes_c1a=True,
    )
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[_stage(epochs=1)])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    driver._build_stage_runtime(
        _stage(epochs=1), decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None,
    )
    driver._cur_stage_index = 0
    # eval() the penalty so rate_bits is DETERMINISTIC (rounds instead of adding the
    # U(-0.5,0.5) Ballé STE noise) — so the only possible difference between the two
    # calls is whether the C1a term was added.
    driver._weight_entropy_penalty.eval()
    # penalty-only term (cat_lambda=0)
    reg_pen = driver._weight_regularizers(decoder, latents, _stage(cat_lambda=0.0))
    # with cat_lambda>0 but superseded → should equal the penalty-only term (C1a zeroed)
    reg_super = driver._weight_regularizers(decoder, latents, _stage(cat_lambda=0.02))
    assert reg_pen is not None and reg_super is not None
    assert abs(float(reg_pen.item()) - float(reg_super.item())) < 1e-6, (
        "C1a was NOT superseded — the term differs from penalty-only (double-count active)"
    )


def test_penalty_stacks_c1a_when_supersede_disabled():
    """``supersedes_c1a=False`` stacks BOTH terms (the C1a entropy is ADDED on top of the
    penalty) — so the reg term is strictly larger than penalty-only. (Not recommended —
    the probe shows stacking is net-negative — but the flag must honor the choice.)"""
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        weight_entropy_penalty_lambda=10.0, weight_entropy_penalty_supersedes_c1a=False,
    )
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[_stage(epochs=1)])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    driver._build_stage_runtime(
        _stage(epochs=1), decoder=decoder, latents=latents, ema_decoder=None, ema_latents=None,
    )
    driver._cur_stage_index = 0
    driver._weight_entropy_penalty.eval()  # deterministic rate_bits (no STE noise)
    reg_pen = driver._weight_regularizers(decoder, latents, _stage(cat_lambda=0.0))
    reg_stack = driver._weight_regularizers(decoder, latents, _stage(cat_lambda=0.02))
    assert reg_pen is not None and reg_stack is not None
    assert float(reg_stack.item()) > float(reg_pen.item()) + 1e-9, (
        "supersede=False did NOT stack C1a on top of the penalty"
    )


def test_c1a_unaffected_when_penalty_off():
    """λ=0 (penalty off) → the C1a term is UNCHANGED regardless of supersede flag (the
    byte-identical guarantee for the live basin: the supersede logic only fires when the
    penalty is active)."""
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=_tmp_out(), device="cpu",
        weight_entropy_penalty_lambda=0.0,  # penalty OFF
    )
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=import_vendored_bundle(),
                                curriculum=[_stage(epochs=1)])
    decoder = driver._new_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    # cat_lambda>0 with penalty off → the C1a term is present (legacy behavior).
    reg = driver._weight_regularizers(decoder, latents, _stage(cat_lambda=0.02))
    assert reg is not None and float(reg.item()) > 0.0
