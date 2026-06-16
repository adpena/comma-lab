# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the configurable-taper carrier (the #1 structural lever).

The defining contract is the PARITY GATE: with the default taper this decoder is a
FAITHFUL GENERALIZATION of the vendored PR95 decoder — same param shapes, bit-identical
forward given the same weights — so the schedule-agnostic vendored codec round-trips a
d_seg-aware taper unchanged. Tests assert REAL behavior: cross-loading vendored weights,
bit-identical forward, actual param counts, the codec round-trip, and the Muon partition —
never constants.
"""
from __future__ import annotations

import pytest
import torch

from tac.torch_vehicle.configurable_taper_decoder import (
    ConfigurableTaperHNeRVDecoder,
    decoder_param_count,
    dseg_aware_taper,
    vendored_taper,
)


def _vendored(base_channels=20):
    from tac.torch_vehicle.driver import import_vendored_bundle

    v = import_vendored_bundle()
    return v.HNeRVDecoder(latent_dim=28, base_channels=base_channels, eval_size=(384, 512))


# ---- the taper-schedule helpers --------------------------------------------
def test_vendored_taper_matches_hardcoded_schedule():
    """vendored_taper(C) == the exact vendored [C,C,C,.75C,.58C,.5C,.5C]."""
    for C in (20, 36, 64):
        v = _vendored(C)
        assert vendored_taper(C) == list(v.channels), f"mismatch at C={C}"


def test_decoder_param_count_is_exact():
    """decoder_param_count must equal the ACTUAL nn parameter count (not an estimate)."""
    for channels in (vendored_taper(20), vendored_taper(36), dseg_aware_taper(20)):
        d = ConfigurableTaperHNeRVDecoder(latent_dim=28, channels=channels)
        actual = sum(p.numel() for p in d.parameters())
        assert decoder_param_count(channels, latent_dim=28) == actual, (
            f"param-count formula wrong for {channels}: predicted "
            f"{decoder_param_count(channels)} != actual {actual}"
        )


def test_dseg_aware_taper_is_byte_matched_and_late_weighted():
    """The d_seg-aware taper reallocates to the mid-late stages and stays ~byte-matched
    (within 5%) to the baseline param count — so an A/B isolates ALLOCATION from capacity."""
    base = vendored_taper(20)
    aware = dseg_aware_taper(20)
    assert len(aware) == 7
    pb, pa = decoder_param_count(base), decoder_param_count(aware)
    assert abs(pa - pb) / pb < 0.05, f"taper not byte-matched: base {pb} vs aware {pa}"
    # late mid-stages (3,4) raised vs baseline; early stages (0,1) lowered.
    assert aware[3] > base[3] and aware[4] > base[4], f"mid-late not raised: {aware} vs {base}"
    assert aware[0] < base[0] and aware[1] < base[1], f"early not lowered: {aware} vs {base}"


# ---- the PARITY GATE (faithful generalization) -----------------------------
def test_default_taper_is_shape_identical_to_vendored():
    """Default taper → identical state_dict keys + shapes as the vendored decoder."""
    v = _vendored(20)
    d = ConfigurableTaperHNeRVDecoder(latent_dim=28, base_channels=20, channels=None)
    vsd, dsd = v.state_dict(), d.state_dict()
    assert set(vsd) == set(dsd), f"key mismatch: {set(vsd) ^ set(dsd)}"
    for k in vsd:
        assert vsd[k].shape == dsd[k].shape, f"shape mismatch at {k}: {vsd[k].shape} vs {dsd[k].shape}"


def test_default_taper_bit_identical_forward_given_same_weights():
    """THE parity contract: load vendored weights into the default-taper decoder → the
    forward output is BIT-IDENTICAL. Proves architectural equivalence (the codec loads
    weights, so same-weights-same-output is the real contract)."""
    torch.manual_seed(0)
    v = _vendored(20).eval()
    d = ConfigurableTaperHNeRVDecoder(latent_dim=28, base_channels=20, channels=None).eval()
    d.load_state_dict(v.state_dict())  # strict — must succeed
    z = torch.randn(4, 28)
    with torch.no_grad():
        ov, od = v(z), d(z)
    assert ov.shape == od.shape == (4, 2, 3, 384, 512)
    assert torch.equal(ov, od), f"forward not bit-identical (max|Δ|={ (ov-od).abs().max():.3e})"


def test_dseg_taper_forward_shape_and_finite():
    """A d_seg-aware taper produces the correct (B,2,3,384,512) output, finite, in [0,255]."""
    d = ConfigurableTaperHNeRVDecoder(latent_dim=28, channels=dseg_aware_taper(20)).eval()
    z = torch.randn(2, 28)
    with torch.no_grad():
        out = d(z)
    assert out.shape == (2, 2, 3, 384, 512)
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 255.0


# ---- the schedule-AGNOSTIC codec round-trip --------------------------------
def test_codec_round_trips_a_dseg_taper():
    """build_archive/parse_archive (the vendored codec) round-trips a NON-default taper's
    state_dict unchanged → the export pipeline is schedule-agnostic (the key enabler)."""
    from tac.torch_vehicle.driver import import_vendored_bundle

    vb = import_vendored_bundle()
    channels = dseg_aware_taper(20)
    d = ConfigurableTaperHNeRVDecoder(latent_dim=28, channels=channels).eval()
    latents = torch.randn(8, 28)
    meta = {"base_channels": 20, "latent_dim": 28, "channels": channels}
    archive = vb.build_archive(d.state_dict(), latents, meta)
    assert isinstance(archive, (bytes, bytearray)) and len(archive) > 0
    dec_sd, lat2, meta2 = vb.parse_archive(archive)
    # The schedule-AGNOSTIC claim: the parsed state_dict has the taper's keys+shapes and
    # loads STRICT into a fresh taper decoder (the structural round-trip). The codec is
    # LOSSY by design (int8-quantizes weights), so the forward matches within the quant
    # tolerance (~sub-0.1 on the 0-255 scale), NOT bit-identically.
    d2 = ConfigurableTaperHNeRVDecoder(latent_dim=28, channels=channels).eval()
    d2.load_state_dict(dec_sd)  # strict — proves keys+shapes round-trip for the non-default taper
    z = torch.randn(2, 28)
    with torch.no_grad():
        out, out2 = d(z), d2(z)
    assert out2.shape == out.shape == (2, 2, 3, 384, 512)
    # within int8-quant tolerance on the 0-255 scale (empirically ~0.05 max abs).
    assert (out - out2).abs().max() < 2.0, (
        f"codec round-trip drifted beyond int8 quant tolerance: max|Δ|={(out-out2).abs().max():.3f}"
    )
    # a meta key round-trips (the codec preserves the meta dict).
    assert meta2.get("base_channels") == 20


def test_muon_partition_works_on_taper_decoder():
    """partition_params_for_muon (splits by NAME) works on the taper decoder: non-empty
    Muon (2D conv weights) + AdamW (stem/rgb/bias) groups, covering all params."""
    from tac.torch_vehicle.driver import import_vendored_bundle

    vb = import_vendored_bundle()
    d = ConfigurableTaperHNeRVDecoder(latent_dim=28, channels=dseg_aware_taper(20))
    muon, adamw = vb.partition_params_for_muon(d)
    assert len(muon) > 0 and len(adamw) > 0
    total = sum(p.numel() for p in muon) + sum(p.numel() for p in adamw)
    assert total == sum(p.numel() for p in d.parameters()), "partition dropped params"


# ---- driver wire-in (default-OFF byte-identical) ---------------------------
def _cfg(**kw):
    from tac.torch_vehicle.driver import TorchVehicleConfig

    base = dict(base_channels=20, latent_dim=28, device="cpu")
    base.update(kw)
    return TorchVehicleConfig(**base)


def test_cfg_taper_default_none():
    assert _cfg().taper_channels is None


def test_cfg_taper_validation():
    with pytest.raises(ValueError, match="7 stages"):
        _cfg(taper_channels=[20, 20, 20])
    with pytest.raises(ValueError, match="positive"):
        _cfg(taper_channels=[20, 20, 20, 15, 11, 0, 10])
    with pytest.raises(ValueError, match="pose_film_enabled=False"):
        _cfg(taper_channels=vendored_taper(20), pose_film_enabled=True)


def test_driver_builds_taper_decoder_when_set(tmp_path):
    from tac.torch_vehicle.driver import TorchVehicleDriver, import_vendored_bundle
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext

    channels = dseg_aware_taper(20)
    cfg = _cfg(out_dir=tmp_path / "t", taper_channels=channels, seed=0)
    drv = TorchVehicleDriver(cfg, scorer=SyntheticScorerContext(n_pairs=4, device="cpu", seed=0),
                             vendored=import_vendored_bundle(), curriculum=[])
    dec = drv._new_vendored_decoder()
    assert isinstance(dec, ConfigurableTaperHNeRVDecoder)
    assert dec.channels == channels


def test_driver_builds_vendored_when_taper_none(tmp_path):
    from tac.torch_vehicle.driver import TorchVehicleDriver, import_vendored_bundle
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext

    cfg = _cfg(out_dir=tmp_path / "v", taper_channels=None, seed=0)
    drv = TorchVehicleDriver(cfg, scorer=SyntheticScorerContext(n_pairs=4, device="cpu", seed=0),
                             vendored=import_vendored_bundle(), curriculum=[])
    dec = drv._new_vendored_decoder()
    # the vendored decoder is NOT our class (byte-identical default path).
    assert not isinstance(dec, ConfigurableTaperHNeRVDecoder)
    assert list(dec.channels) == vendored_taper(20)


# ---- resume-safety: taper persisted in manifest + fail-closed on mismatch (F fix) ----
def _short_taper_curriculum(epochs=3):
    import torch.nn.functional as F

    from tac.torch_vehicle.curriculum import StageSpec

    def ce(s, t):
        return F.cross_entropy(s, t)

    return [StageSpec(
        name="t", epochs=epochs, seg_loss_fn=ce, eval_every=epochs, batch_size=4, ema_decay=0.999,
        use_muon=False, adamw_lr=1e-3, muon_lr=2e-4, muon_weight_decay=0.0, latent_lr_mult=10.0,
        grad_clip=1e9, grad_clip_muon=1e9, lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0,
        cat_lambda=0.0, cat_sigma=0.2, use_qat=False, init_latents_random=True,
    )]


def _taper_driver(out_dir, channels):
    from tac.torch_vehicle.driver import TorchVehicleDriver, import_vendored_bundle
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext

    cfg = _cfg(out_dir=out_dir, taper_channels=channels, seed=0, checkpoint_every_epochs=1)
    return TorchVehicleDriver(cfg, scorer=SyntheticScorerContext(n_pairs=4, device="cpu", seed=0),
                              vendored=import_vendored_bundle(), curriculum=_short_taper_curriculum(3))


def test_manifest_persists_taper_channels(tmp_path):
    """The checkpoint manifest records taper_channels (so resume can verify it)."""
    from tac.torch_vehicle.checkpoint import read_manifest
    from tac.torch_vehicle.driver import _SimulatedDeath

    ch = dseg_aware_taper(20)
    d = _taper_driver(tmp_path / "r", ch)
    d._stop_after_global_epoch = 1  # land a checkpoint, then die before DONE
    with pytest.raises(_SimulatedDeath):
        d.run()
    man = read_manifest(tmp_path / "r")
    assert man.get("taper_channels") == ch, f"manifest missing/wrong taper: {man.get('taper_channels')}"


def test_resume_with_different_taper_fails_closed(tmp_path):
    """Resuming an out_dir trained with taper A using a DIFFERENT taper B raises EXPLICITLY
    (not via an accidental shape mismatch) — the resume-safety contract."""
    from tac.torch_vehicle.driver import _SimulatedDeath

    a = _taper_driver(tmp_path / "x", dseg_aware_taper(20))
    a._stop_after_global_epoch = 1
    with pytest.raises(_SimulatedDeath):
        a.run()
    # fresh driver, DIFFERENT taper, same out_dir → run() must refuse on the taper guard.
    b = _taper_driver(tmp_path / "x", [18, 18, 18, 16, 12, 10, 10])
    with pytest.raises(ValueError, match="cannot resume a different taper"):
        b.run()


def test_resume_with_same_taper_succeeds(tmp_path):
    """Resuming the SAME taper continues cleanly (the guard does not false-positive)."""
    from tac.torch_vehicle.driver import _SimulatedDeath

    ch = dseg_aware_taper(20)
    a = _taper_driver(tmp_path / "y", ch)
    a._stop_after_global_epoch = 1
    with pytest.raises(_SimulatedDeath):
        a.run()
    b = _taper_driver(tmp_path / "y", ch)
    summ = b.run()  # must NOT raise; completes the 3-epoch stage
    assert (tmp_path / "y" / "torch_vehicle_run.DONE").exists()
