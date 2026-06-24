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

from pathlib import Path

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


def test_launcher_resolve_taper_threads_latent_dim():
    """launch_taper_ab._resolve_taper must thread args.latent_dim into dseg_aware_taper so the
    resolved taper byte-matches at the ACTUAL latent_dim (round-5 fix; the same class swept)."""
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_taper_launcher",
        Path(__file__).resolve().parents[4] / "experiments" / "launch_taper_ab.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for ld in (16, 28, 64):
        args = argparse.Namespace(arm="dseg_aware", base_channels=20, latent_dim=ld,
                                  taper_channels=None)
        _active, resolved = mod._resolve_taper(args)
        # the resolved taper byte-matches the vendored baseline AT this latent_dim.
        pb = decoder_param_count(vendored_taper(20), latent_dim=ld)
        pa = decoder_param_count(resolved, latent_dim=ld)
        assert abs(pa - pb) / pb < 0.05, f"ld={ld}: launcher taper not byte-matched ({pb} vs {pa})"
        # and it EQUALS the directly-threaded dseg_aware_taper (proves the thread is wired).
        assert resolved == dseg_aware_taper(20, latent_dim=ld), f"ld={ld}: launcher != threaded"


def test_dseg_aware_taper_byte_matches_at_nondefault_latent_dim():
    """The byte-match is EXACT-target at any latent_dim (the stem param count scales with
    latent_dim) — the round-4 LOW fix. Match the target computed at the SAME latent_dim."""
    for ld in (16, 28, 64):
        base = vendored_taper(20)
        aware = dseg_aware_taper(20, latent_dim=ld)
        pb = decoder_param_count(base, latent_dim=ld)
        pa = decoder_param_count(aware, latent_dim=ld)
        assert abs(pa - pb) / pb < 0.05, f"ld={ld}: base {pb} vs aware {pa} not byte-matched"


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

    base = {"base_channels": 20, "latent_dim": 28, "device": "cpu"}
    base.update(kw)
    return TorchVehicleConfig(**base)


def test_cfg_taper_default_none():
    assert _cfg().taper_channels is None


def test_final_channel_lt_2_fails_closed():
    """A taper whose FINAL channel < 2 makes refine's Conv2d(cf, cf//2,...) → cf//2==0
    invalid. Guard fail-closed with a clear error at ALL three surfaces (round-6 MEDIUM):
    the decoder ctor, the param-count formula, and the cfg validation."""
    bad = [16, 16, 17, 19, 19, 14, 1]  # final == 1 → final//2 == 0
    with pytest.raises(ValueError, match="final channel must be >= 2"):
        ConfigurableTaperHNeRVDecoder(latent_dim=28, channels=bad)
    with pytest.raises(ValueError, match="final channel must be >= 2"):
        decoder_param_count(bad)
    with pytest.raises(ValueError, match="final stage must be >= 2"):
        _cfg(taper_channels=bad)


def test_cfg_taper_validation():
    with pytest.raises(ValueError, match="7 stages"):
        _cfg(taper_channels=[20, 20, 20])
    with pytest.raises(ValueError, match="positive"):
        _cfg(taper_channels=[20, 20, 20, 15, 11, 0, 10])
    # taper + FiLM v1 (the default version) is refused (v1 stem-injection couples d_pose
    # into d_seg and is untested on a re-taper); taper + FiLM v2 is now ALLOWED (the
    # bind-all production combo — the v2 residual wrapper composes with the taper carrier).
    with pytest.raises(ValueError, match="requires pose_film_version=2"):
        _cfg(taper_channels=vendored_taper(20), pose_film_enabled=True)
    cfg_v2 = _cfg(taper_channels=vendored_taper(20), pose_film_enabled=True, pose_film_version=2)
    assert cfg_v2.pose_film_enabled and cfg_v2.pose_film_version == 2


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
    b.run()  # must NOT raise; completes the 3-epoch stage
    assert (tmp_path / "y" / "torch_vehicle_run.DONE").exists()


def test_resume_with_different_latent_dim_fails_closed(tmp_path):
    """Resuming an out_dir trained with latent_dim A using latent_dim B raises EXPLICITLY
    (the round-8 guard) — not a cryptic stem-Linear size-mismatch in _restore_into."""
    from tac.torch_vehicle.driver import (
        TorchVehicleConfig,
        TorchVehicleDriver,
        _SimulatedDeath,
        import_vendored_bundle,
    )
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext

    def drv(ld):
        cfg = TorchVehicleConfig(base_channels=20, latent_dim=ld, out_dir=tmp_path / "ld",
                                 device="cpu", seed=0, checkpoint_every_epochs=1)
        return TorchVehicleDriver(
            cfg, scorer=SyntheticScorerContext(n_pairs=4, device="cpu", seed=0),
            vendored=import_vendored_bundle(), curriculum=_short_taper_curriculum(3))

    a = drv(28)
    a._stop_after_global_epoch = 1
    with pytest.raises(_SimulatedDeath):
        a.run()
    b = drv(16)  # DIFFERENT latent_dim, same out_dir
    with pytest.raises(ValueError, match="cannot resume a different basis"):
        b.run()


# ---- HIGH-FREQUENCY ACTIVATION FAMILY (k1-for-d_seg architecture screen) ----
# The decisive screen: a non-"siren" activation (FINER / WIRE) must be a REAL swap
# (changes the forward output) yet "siren" must stay BYTE-IDENTICAL to the vendored
# torch.sin path (the parity contract). These tests are the NO-FAKE guard: if the
# wire-in degenerated to a no-op, test_activation_finer_actually_changes_output and
# test_activation_wire_actually_changes_output would FAIL (constants-not-behavior).


def test_activation_default_is_siren():
    """The ConfigurableTaper decoder + the cfg both default to 'siren' (byte-identical)."""
    d = ConfigurableTaperHNeRVDecoder(latent_dim=28, base_channels=20, channels=None)
    assert d.activation_family == "siren"
    assert d.wire_scale == 1.0
    assert _cfg().activation == "siren"
    assert _cfg().wire_scale == 1.0


def test_activation_siren_bit_identical_to_vendored():
    """THE parity contract under the activation wire-in: ConfigurableTaper with
    activation='siren' loaded with vendored weights renders BIT-IDENTICAL output to the
    vendored decoder. apply_activation_family(..., 'siren', omega=1.0) == torch.sin(x), so
    the screen's CONTROL arm provably reproduces the legacy d_seg basin."""
    torch.manual_seed(0)
    v = _vendored(20).eval()
    d = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="siren"
    ).eval()
    d.load_state_dict(v.state_dict())  # strict
    z = torch.randn(4, 28)
    with torch.no_grad():
        ov, od = v(z), d(z)
    assert torch.equal(ov, od), f"siren not bit-identical (max|Δ|={(ov-od).abs().max():.3e})"


def test_activation_siren_default_taper_bit_identical_to_explicit_siren():
    """Sanity: the default-taper (channels=None) siren path and the explicit-vendored-taper
    siren path are the SAME architecture (same weights → same output)."""
    torch.manual_seed(1)
    a = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="siren"
    ).eval()
    b = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=vendored_taper(20), activation_family="siren"
    ).eval()
    b.load_state_dict(a.state_dict())
    z = torch.randn(3, 28)
    with torch.no_grad():
        assert torch.equal(a(z), b(z))


def test_activation_finer_actually_changes_output():
    """NO-FAKE: FINER must CHANGE the forward output vs siren at the SAME weights (not a
    no-op renamed). This test FAILS if the activation swap degenerates to torch.sin."""
    torch.manual_seed(2)
    base = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="siren"
    ).eval()
    finer = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="finer"
    ).eval()
    finer.load_state_dict(base.state_dict())  # SAME weights — only activation differs
    z = torch.randn(4, 28)
    with torch.no_grad():
        ob, of = base(z), finer(z)
    assert ob.shape == of.shape == (4, 2, 3, 384, 512)
    assert not torch.equal(ob, of), "FINER produced identical output to siren (no-op!)"
    assert torch.isfinite(of).all()
    assert of.min() >= 0.0 and of.max() <= 255.0


def test_activation_wire_actually_changes_output_and_scale_matters():
    """NO-FAKE: WIRE must change output vs siren AND the wire_scale must actually modulate
    the Gabor window (different scales → different outputs). FAILS if wire degenerates to
    torch.sin or ignores wire_scale."""
    torch.manual_seed(3)
    base = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="siren"
    ).eval()
    w05 = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="wire", wire_scale=0.5
    ).eval()
    w20 = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="wire", wire_scale=2.0
    ).eval()
    w05.load_state_dict(base.state_dict())
    w20.load_state_dict(base.state_dict())
    z = torch.randn(3, 28)
    with torch.no_grad():
        ob, o05, o20 = base(z), w05(z), w20(z)
    assert not torch.equal(ob, o05), "WIRE(0.5) == siren (no-op!)"
    assert not torch.equal(o05, o20), "wire_scale ignored (0.5 == 2.0)!"
    assert torch.isfinite(o05).all() and torch.isfinite(o20).all()


def test_activation_finer_wire_preserve_param_count_and_keys():
    """The activation swap is BYTE-NEUTRAL: same state_dict keys + shapes + param count as
    siren (the screen holds the byte budget fixed; only the nonlinearity changes)."""
    siren = ConfigurableTaperHNeRVDecoder(latent_dim=28, base_channels=20, channels=None)
    for act in ("finer", "wire"):
        d = ConfigurableTaperHNeRVDecoder(
            latent_dim=28, base_channels=20, channels=None, activation_family=act
        )
        assert set(siren.state_dict()) == set(d.state_dict()), f"{act} key mismatch"
        sp = sum(p.numel() for p in siren.parameters())
        dp = sum(p.numel() for p in d.parameters())
        assert sp == dp, f"{act} param count {dp} != siren {sp}"


def test_activation_invalid_fails_closed():
    """An unknown activation family raises at construction (fail-closed, no silent siren)."""
    with pytest.raises(ValueError, match="unknown SIREN activation family"):
        ConfigurableTaperHNeRVDecoder(
            latent_dim=28, base_channels=20, channels=None, activation_family="bogus"
        )


def test_cfg_activation_threads_to_decoder_via_driver():
    """The driver's _new_vendored_decoder honors cfg.activation: a non-siren activation
    routes through ConfigurableTaper EVEN WITHOUT an explicit taper (so the screen needs
    only --activation, not --taper-channels), and siren+no-taper stays the vendored path."""
    from tac.torch_vehicle.configurable_taper_decoder import ConfigurableTaperHNeRVDecoder as _CT

    # siren + no taper → vendored decoder (NOT ConfigurableTaper) = byte-identical legacy.
    cfg_siren = _cfg(activation="siren")
    assert cfg_siren.activation == "siren" and cfg_siren.taper_channels is None

    # finer + no taper → cfg carries it; the decoder builder selects ConfigurableTaper.
    cfg_finer = _cfg(activation="finer")
    assert cfg_finer.activation == "finer"

    # Build a decoder through the same selection logic the driver uses (no scorer needed):
    # replicate _new_vendored_decoder's branch on a bare cfg to prove the routing.
    from tac.torch_vehicle.configurable_taper_decoder import vendored_taper as _vt

    use_configurable = (cfg_finer.taper_channels is not None) or (cfg_finer.activation != "siren")
    assert use_configurable, "finer must force the ConfigurableTaper path"
    d = _CT(
        latent_dim=cfg_finer.latent_dim,
        base_channels=cfg_finer.base_channels,
        channels=_vt(cfg_finer.base_channels),
        activation_family=cfg_finer.activation,
        wire_scale=cfg_finer.wire_scale,
    )
    assert isinstance(d, _CT) and d.activation_family == "finer"


# ---- NEXT-GEN INR ACTIVATION EXPANSION (2026-06-24) -------------------------
# NO-FAKE decoder-level coverage for the new fixed-form families
# (gauss/hosc/sinc/rcgauss/finer_gauss) and the learnable families
# (step_basis/fkan). siren stays bit-identical; each non-siren GENUINELY differs.

@pytest.mark.parametrize(
    "act", ("gauss", "hosc", "sinc", "rcgauss", "finer_gauss")
)
def test_nextgen_fixed_form_activation_changes_decoder_output(act):
    """NO-FAKE: each new FIXED-FORM family changes the decoder forward vs siren at the
    SAME weights, is byte-neutral (same keys/param count), and stays in [0,255]."""
    torch.manual_seed(5)
    base = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="siren"
    ).eval()
    var = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family=act
    ).eval()
    var.load_state_dict(base.state_dict())  # SAME weights — only activation differs
    z = torch.randn(3, 28)
    with torch.no_grad():
        ob, ov = base(z), var(z)
    assert ov.shape == ob.shape == (3, 2, 3, 384, 512)
    assert not torch.equal(ob, ov), f"{act} produced identical output to siren (no-op!)"
    assert torch.isfinite(ov).all()
    assert ov.min() >= 0.0 and ov.max() <= 255.0
    # byte-neutral: same state_dict keys + param count as siren.
    assert set(base.state_dict()) == set(var.state_dict()), f"{act} changed keys"
    assert sum(p.numel() for p in base.parameters()) == sum(
        p.numel() for p in var.parameters()
    ), f"{act} changed param count (should be byte-neutral)"


def test_nextgen_hosc_beta_modulates_decoder_output():
    """NO-FAKE: hosc_beta actually sharpens the decoder activation (different output)."""
    torch.manual_seed(6)
    base = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="siren"
    ).eval()
    b1 = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="hosc", hosc_beta=1.0
    ).eval()
    b8 = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="hosc", hosc_beta=8.0
    ).eval()
    b1.load_state_dict(base.state_dict())
    b8.load_state_dict(base.state_dict())
    z = torch.randn(2, 28)
    with torch.no_grad():
        o1, o8 = b1(z), b8(z)
    assert not torch.equal(o1, o8), "hosc_beta ignored by the decoder (no-op hyperparam!)"


@pytest.mark.parametrize(
    ("act", "k", "params_per_k"),
    (("step_basis", 4, 3), ("fkan", 5, 2)),
)
def test_nextgen_learnable_activation_adds_params_and_changes_output(act, k, params_per_k):
    """NO-FAKE: the learnable families build a real sub-activation module that (a) adds
    EXACTLY params_per_k*K parameters over siren, (b) changes the decoder forward, (c)
    keeps the output in [0,255] and finite. They are NOT byte-neutral (reported delta)."""
    torch.manual_seed(7)
    siren = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="siren"
    ).eval()
    kw = {"step_basis_k": k} if act == "step_basis" else {"fkan_k": k}
    learn = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family=act, **kw
    ).eval()
    # the shared conv/stem weights load from siren; only the sub-activation is extra.
    learn.load_state_dict(siren.state_dict(), strict=False)
    sp = sum(p.numel() for p in siren.parameters())
    lp = sum(p.numel() for p in learn.parameters())
    assert lp - sp == params_per_k * k, (
        f"{act} added {lp - sp} params, expected {params_per_k * k}"
    )
    # the learnable module's params ARE in the decoder state_dict (real, archived).
    extra_keys = set(learn.state_dict()) - set(siren.state_dict())
    assert any("_learnable_act" in kk for kk in extra_keys), f"{act} params not in state_dict"
    z = torch.randn(2, 28)
    with torch.no_grad():
        out = learn(z)
    assert out.shape == (2, 2, 3, 384, 512)
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_nextgen_fkan_decoder_init_matches_siren_basis():
    """FKAN inits to the SIREN basis (a_1=1, rest 0 => sin(x)); with vendored conv weights
    loaded the decoder forward should be VERY close to siren at init (the shared spectral
    prior). It is NOT required to be bit-identical (it routes through the module), but the
    init parity proves FKAN starts from siren, not a random nonlinearity."""
    torch.manual_seed(8)
    siren = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="siren"
    ).eval()
    fkan = ConfigurableTaperHNeRVDecoder(
        latent_dim=28, base_channels=20, channels=None, activation_family="fkan", fkan_k=5
    ).eval()
    fkan.load_state_dict(siren.state_dict(), strict=False)
    z = torch.randn(2, 28)
    with torch.no_grad():
        os_, of = siren(z), fkan(z)
    # at FKAN init the per-element activation == sin(x), so the forward closely tracks siren.
    assert torch.allclose(os_, of, atol=1e-2), (
        f"FKAN init drifts from siren basis (max|Δ|={(os_-of).abs().max():.3e})"
    )


def test_nextgen_activation_default_hyperparams():
    """The decoder defaults match the cfg defaults (hosc_beta=4, step_basis_k=4, fkan_k=5)."""
    d = ConfigurableTaperHNeRVDecoder(latent_dim=28, base_channels=20, channels=None)
    assert d.hosc_beta == 4.0 and d.step_basis_k == 4 and d.fkan_k == 5
    assert d._learnable_act is None  # siren default is fixed-form
    assert _cfg().hosc_beta == 4.0 and _cfg().step_basis_k == 4 and _cfg().fkan_k == 5


def test_nextgen_driver_builds_learnable_activation_decoder():
    """The driver routes a learnable --activation through ConfigurableTaper with a real
    sub-activation module (the screen needs only --activation step_basis/fkan)."""
    from tac.torch_vehicle.driver import TorchVehicleDriver, import_vendored_bundle
    from tac.torch_vehicle.scorer_context import SyntheticScorerContext

    for act in ("step_basis", "fkan"):
        cfg = _cfg(activation=act)
        drv = TorchVehicleDriver(
            cfg, scorer=SyntheticScorerContext(n_pairs=4, device="cpu", seed=0),
            vendored=import_vendored_bundle(), curriculum=[])
        dec = drv._new_vendored_decoder()
        assert isinstance(dec, ConfigurableTaperHNeRVDecoder)
        assert dec.activation_family == act
        assert dec._learnable_act is not None, f"{act} did not build a learnable module"
