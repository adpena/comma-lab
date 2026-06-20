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
