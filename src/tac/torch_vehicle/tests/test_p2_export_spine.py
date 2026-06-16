# SPDX-License-Identifier: MIT
"""P2 export wiring + the SINGLE-SOURCE SENSITIVITY SPINE (the synergistic core).

These tests guard the production EXPORT path the bind-all build-out depends on:

* The SPINE RECONCILIATION (the bug the P2 fix closes): the online score-sensitivity
  EMA is keyed by ``decoder.named_modules()`` MODULE names (``blocks.0``) — the SAME
  keys ``apply_score_aware_qat`` (the training-time QAT bits) consumes — but the
  variable-level EXPORT codec checks the WEIGHT state-dict keys (``blocks.0.weight``).
  Before the fix, feeding the production module-keyed EMA into the export silently
  no-op'd the rate-attack (uniform -> vendored byte-identical) because the codec lookup
  missed every EMA entry. ``_sensitivity_for_codec_weight_keys`` rebinds the EMA so the
  SAME ``||dS/dw_t||`` tensor drives BOTH the QAT grid AND the export grid. The
  single-source spine is the synergistic core: capacity (taper) + bits (QAT) + levels
  (codec) all follow ONE d_seg-geometry.

* The PRODUCTION EXPORT method (``export_production_archive``): rate-attack (decoder
  blob) + L3 finishing-kit (PR98/T10/S12 trailing section) composed into ONE byte-
  closed packet, default-preserving (both levers off -> byte-identical to vendored).

NO-FAKE: every byte delta is a REAL measurement on a real decoder state dict; the
default-OFF paths are asserted byte-identical; the spine fix is proven by a
behavioral contrast (module-keyed EMA must produce the SAME levels as the equivalent
codec-keyed EMA, and must STRICTLY shrink the archive — a no-op spine fails).

Authority: ``[contest-CPU advisory]`` — NO score claim (the exact d_seg/d_pose that
pick BEST still run the full scorer on the CPU authority on the byte-closed archive).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tac.torch_vehicle.distortion_finishing_kit import DistortionKitConfig
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    _sensitivity_for_codec_weight_keys,
    _split_three_section_archive,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _module_weight_keys(decoder) -> tuple[list[str], list[str]]:
    """Return (module_names, sd_weight_keys) for the conv/linear tensors — the two
    DISTINCT namespaces the spine must reconcile."""
    mods = [
        n
        for n, m in decoder.named_modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        and getattr(m, "weight", None) is not None
    ]
    wkeys = [k for k in decoder.state_dict() if k.endswith(".weight")]
    return mods, wkeys


def _drv(cfg_kwargs: dict, *, n_pairs: int = 6) -> tuple[TorchVehicleDriver, object]:
    v = import_vendored_bundle()
    sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=0)
    kwargs = {"base_channels": 20, "latent_dim": 28, "device": "cpu", **cfg_kwargs}
    cfg = TorchVehicleConfig(**kwargs)
    return TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[]), v


# ---------------------------------------------------------------------------
# Part 1 — the SPINE RECONCILIATION helper (the namespace bridge)
# ---------------------------------------------------------------------------
def test_spine_helper_rekeys_module_keyed_ema_onto_codec_weight_keys() -> None:
    """The EMA convention (module names) maps onto the codec convention (weight keys)."""
    v = import_vendored_bundle()
    dec = v.HNeRVDecoder(latent_dim=28, base_channels=20)
    mods, wkeys = _module_weight_keys(dec)
    # The two namespaces are genuinely DISJOINT (the bug's root cause).
    assert set(mods).isdisjoint(set(wkeys))
    module_sens = {n: float(i + 1) for i, n in enumerate(mods)}
    rekeyed = _sensitivity_for_codec_weight_keys(module_sens, wkeys)
    assert rekeyed is not None
    # EVERY codec weight key resolved to its module's EMA value.
    assert set(rekeyed) == set(wkeys)
    for n, wk in zip(mods, [m + ".weight" for m in mods], strict=True):
        assert rekeyed[wk] == module_sens[n]


def test_spine_helper_handles_film_wrapper_prefix_and_codec_keys() -> None:
    """The helper resolves the FiLM-wrapper EMA prefix (``decoder.<module>``) AND the
    already-codec-keyed test convention (``<module>.weight``)."""
    wkeys = ["stem.weight", "blocks.0.weight"]
    # FiLM EMA convention: keys prefixed ``decoder.``.
    film_sens = {"decoder.stem": 3.0, "decoder.blocks.0": 7.0}
    rk = _sensitivity_for_codec_weight_keys(film_sens, wkeys)
    assert rk == {"stem.weight": 3.0, "blocks.0.weight": 7.0}
    # Already-codec-keyed (the test-API convention) passes through unchanged.
    codec_sens = {"stem.weight": 1.0, "blocks.0.weight": 2.0}
    assert _sensitivity_for_codec_weight_keys(codec_sens, wkeys) == codec_sens


def test_spine_helper_none_and_unresolved_return_none_default_preserving() -> None:
    """``None`` in -> ``None`` out; an EMA that resolves to NOTHING -> ``None`` (so the
    codec's uniform/byte-identical default-preserving path is preserved — never a
    fabricated allocation)."""
    assert _sensitivity_for_codec_weight_keys(None, ["stem.weight"]) is None
    # Keys that match no codec weight key resolve to nothing -> None.
    assert _sensitivity_for_codec_weight_keys({"not_a_tensor": 1.0}, ["stem.weight"]) is None


# ---------------------------------------------------------------------------
# Part 2 — the END-TO-END spine: the PRODUCTION module-keyed EMA fires the rate-attack
# (the regression guard for the silent-no-op bug)
# ---------------------------------------------------------------------------
def test_production_module_keyed_ema_fires_rate_attack_end_to_end() -> None:
    """THE REGRESSION GUARD. The production EMA is module-keyed (``blocks.0``). Feeding
    it through ``_build_archive_and_eval_decoder`` MUST strictly shrink the archive (the
    rate-attack fires) — before the spine fix this silently no-op'd (uniform -> vendored
    byte-identical) because the codec missed every module-keyed EMA entry. A no-op spine
    fails this test."""
    drv, v = _drv({"lever4_variable_level_export_enabled": True})
    dec = drv._new_vendored_decoder(device=torch.device("cpu"))
    sd = {k: t.detach().clone() for k, t in dec.state_dict().items()}
    latents = torch.randn(6, 28, generator=torch.Generator().manual_seed(3)) * 0.1
    meta = {"n_pairs": 6, "latent_dim": 28, "base_channels": 20, "eval_size": [384, 512]}
    vendored = v.build_archive(sd, latents, meta_dict=dict(meta))

    mods, _wkeys = _module_weight_keys(dec)
    # MODULE-KEYED sensitivity (the EXACT convention accumulate_tensor_sensitivity emits):
    # lower half score-irrelevant (coarse grid), upper half highly sensitive (kept 127).
    module_sens = {
        n: (1e-6 if i < len(mods) // 2 else 100.0) for i, n in enumerate(mods)
    }
    arc, eval_dec, lat = drv._build_archive_and_eval_decoder(
        sd, latents, dict(meta), sensitivity=module_sens
    )
    assert len(arc) < len(vendored), (
        f"SPINE BROKEN: module-keyed EMA did NOT shrink the archive "
        f"({len(arc)} !< {len(vendored)}) — the rate-attack silently no-op'd because the "
        "codec missed the module-keyed EMA keys (the bug the spine fix closes)"
    )
    # Parse-back faithful: the variable decoder keys match the vendored decoder keys.
    assert set(eval_dec.state_dict().keys()) == set(sd.keys())
    assert lat.shape[0] == 6


def test_spine_module_keyed_ema_matches_codec_keyed_ema_levels() -> None:
    """SINGLE-SOURCE PROOF: the SAME sensitivity expressed in the two namespaces produces
    the IDENTICAL export archive — i.e. the QAT (module-keyed) and the codec (weight-
    keyed) truly read ONE map. A spine that read different maps would diverge here."""
    drv, _v = _drv({"lever4_variable_level_export_enabled": True})
    dec = drv._new_vendored_decoder(device=torch.device("cpu"))
    sd = {k: t.detach().clone() for k, t in dec.state_dict().items()}
    latents = torch.randn(6, 28, generator=torch.Generator().manual_seed(11)) * 0.1
    meta = {"n_pairs": 6, "latent_dim": 28, "base_channels": 20, "eval_size": [384, 512]}
    mods, wkeys = _module_weight_keys(dec)
    vals = [1e-6 if i < len(mods) // 2 else 100.0 for i in range(len(mods))]
    module_sens = dict(zip(mods, vals, strict=True))
    codec_sens = dict(zip([m + ".weight" for m in mods], vals, strict=True))

    arc_mod, _e1, _l1 = drv._build_archive_and_eval_decoder(
        sd, latents, dict(meta), sensitivity=module_sens
    )
    arc_codec, _e2, _l2 = drv._build_archive_and_eval_decoder(
        sd, latents, dict(meta), sensitivity=codec_sens
    )
    assert arc_mod == arc_codec, (
        "the module-keyed EMA and the equivalent codec-keyed EMA produced DIFFERENT "
        "export archives — the spine is not single-source (QAT and codec read different "
        "maps)"
    )


def test_spine_qat_and_codec_levels_derive_from_same_band() -> None:
    """The QAT level map (training bits) and the codec level map (export levels) are the
    SAME function of the SAME sensitivity tensor — the literal single-source contract.
    ``apply_score_aware_qat`` and ``levels_from_sensitivity_for_codec`` both delegate to
    ``per_tensor_levels_from_sensitivity`` over the SAME rank-norm band."""
    from tac.losses.variable_level_codec import levels_from_sensitivity_for_codec
    from tac.torch_vehicle.score_aware_qat import per_tensor_levels_from_sensitivity

    v = import_vendored_bundle()
    dec = v.HNeRVDecoder(latent_dim=28, base_channels=20)
    mods, wkeys = _module_weight_keys(dec)
    vals = [1e-6 if i < len(mods) // 2 else 100.0 for i in range(len(mods))]
    module_sens = dict(zip(mods, vals, strict=True))
    # QAT bits: keyed by module names (the training-time path).
    qat_levels = per_tensor_levels_from_sensitivity(module_sens, mods)
    # Codec levels: keyed by weight keys, fed the RE-KEYED EMA (the spine bridge).
    codec_sens = _sensitivity_for_codec_weight_keys(module_sens, wkeys)
    codec_levels = levels_from_sensitivity_for_codec(codec_sens, wkeys)
    # Same tensor, same band -> same per-tensor level count (mapped through the bridge).
    for m in mods:
        assert qat_levels[m] == codec_levels[m + ".weight"], (
            f"QAT bits and codec levels DIVERGE for {m}: {qat_levels[m]} != "
            f"{codec_levels[m + '.weight']} — not single-source"
        )


# ---------------------------------------------------------------------------
# Part 3 — the PRODUCTION EXPORT method (rate-attack + L3, default-preserving)
# ---------------------------------------------------------------------------
def _basin_or_skip() -> tuple[dict, torch.Tensor, int, int]:
    basin = (
        Path(__file__).resolve().parents[4]
        / "experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best"
    )
    if not (basin / "best_ema_decoder.pt").exists():
        pytest.skip("converged basin checkpoint not present")
    ema_sd = torch.load(basin / "best_ema_decoder.pt", map_location="cpu")
    ema_lat = torch.load(basin / "best_ema_latents.pt", map_location="cpu")
    return ema_sd, ema_lat, int(ema_lat.shape[0]), int(ema_lat.shape[1])


def test_export_production_default_off_is_byte_identical_to_vendored() -> None:
    """Both levers OFF -> the finished archive is byte-identical to the pristine vendored
    build_archive (the daemon-safety / default-preserving contract)."""
    ema_sd, ema_lat, n_pairs, ld = _basin_or_skip()
    with tempfile.TemporaryDirectory() as td:
        drv, v = _drv({"out_dir": Path(td), "latent_dim": ld}, n_pairs=n_pairs)
        meta = {
            "n_pairs": n_pairs,
            "latent_dim": ld,
            "base_channels": 20,
            "eval_size": [384, 512],
        }
        vendored = v.build_archive(ema_sd, ema_lat, meta_dict=dict(meta))
        # Even passing a non-uniform sensitivity, the flag-OFF path ignores it.
        dec = drv._new_vendored_decoder(device=torch.device("cpu"))
        mods, _ = _module_weight_keys(dec)
        sens = {n: float(i + 1) for i, n in enumerate(mods)}
        r = drv.export_production_archive(ema_sd, ema_lat, sensitivity=sens)
    assert r["finished_archive"] == vendored
    assert r["is_byte_identical_to_vendored_base"] is True
    assert r["section_bytes"] == 0
    assert r["rate_attack_enabled"] is False
    assert r["decoder_codec"] == "vendored"
    assert r["score_claim"] is False


def test_export_production_rate_attack_plus_l3_reduces_bytes_real_basin() -> None:
    """Rate-attack ON + L3 PR98 kit ON -> finished archive is STRICTLY smaller than
    vendored (the decoder-blob rate saving dominates the <=54B L3 section), driven by the
    production module-keyed EMA (the spine). MEASURED on the real basin."""
    ema_sd, ema_lat, n_pairs, ld = _basin_or_skip()
    with tempfile.TemporaryDirectory() as td:
        drv, v = _drv(
            {
                "out_dir": Path(td),
                "latent_dim": ld,
                "lever4_variable_level_export_enabled": True,
                "distortion_kit": DistortionKitConfig.from_converged_residual_pr98(),
            },
            n_pairs=n_pairs,
        )
        meta = {
            "n_pairs": n_pairs,
            "latent_dim": ld,
            "base_channels": 20,
            "eval_size": [384, 512],
        }
        vendored = v.build_archive(ema_sd, ema_lat, meta_dict=dict(meta))
        dec = drv._new_vendored_decoder(device=torch.device("cpu"))
        mods, _ = _module_weight_keys(dec)
        module_sens = {
            n: (1e-6 if i < len(mods) // 2 else 100.0) for i, n in enumerate(mods)
        }
        r = drv.export_production_archive(ema_sd, ema_lat, sensitivity=module_sens)
    assert r["finished_archive_bytes"] < len(vendored), (
        "rate-attack+L3 did NOT shrink the archive on the real basin (spine no-op or "
        "L3 section overwhelmed the decoder saving)"
    )
    assert r["section_bytes"] <= 54
    assert r["distortion_section_present"] is True
    assert r["decoder_codec"] == "lever4_sensitivity_variable_level.v1"
    # The net delta is the decoder rate saving (negative) + the L3 section (+<=54B).
    assert r["added_bytes"] == r["section_bytes"]


def test_export_production_finished_packet_splits_back_to_base_and_kit() -> None:
    """The finished packet round-trips: split_finished_archive recovers the base archive
    + the kit, and the base parses back to the decoder + latents (byte-closed faithful)."""
    from tac.torch_vehicle.driver import split_finished_archive

    ema_sd, ema_lat, n_pairs, ld = _basin_or_skip()
    with tempfile.TemporaryDirectory() as td:
        drv, v = _drv(
            {
                "out_dir": Path(td),
                "latent_dim": ld,
                "lever4_variable_level_export_enabled": True,
                "distortion_kit": DistortionKitConfig.from_converged_residual_pr98(),
            },
            n_pairs=n_pairs,
        )
        dec = drv._new_vendored_decoder(device=torch.device("cpu"))
        mods, _ = _module_weight_keys(dec)
        module_sens = {
            n: (1e-6 if i < len(mods) // 2 else 100.0) for i, n in enumerate(mods)
        }
        r = drv.export_production_archive(ema_sd, ema_lat, sensitivity=module_sens)
        base, kit = split_finished_archive(r["finished_archive"])
        assert len(base) == r["base_archive_bytes"]
        assert kit.enabled is True
        # The base (rate-attacked) archive's meta carries the variable-level codec flag
        # so the inflate side dispatches the variable decoder reader.
        import brotli

        meta_brotli, _d, _l, _t = _split_three_section_archive(base)
        import json as _json

        emitted = _json.loads(brotli.decompress(meta_brotli).decode("utf-8"))
        assert emitted.get("decoder_codec") == "lever4_sensitivity_variable_level.v1"


def test_export_production_score_finished_requires_real_scorer() -> None:
    """score_finished=True with the synthetic scorer fails closed (no fake kit-aware
    score) — the kit-aware re-score needs the real distortion_net + video."""
    ema_sd, ema_lat, n_pairs, ld = _basin_or_skip()
    with tempfile.TemporaryDirectory() as td:
        drv, _v = _drv(
            {
                "out_dir": Path(td),
                "latent_dim": ld,
                "distortion_kit": DistortionKitConfig.from_converged_residual_pr98(),
            },
            n_pairs=n_pairs,
        )
        with pytest.raises(ValueError, match="real scorer context"):
            drv.export_production_archive(
                ema_sd, ema_lat, sensitivity=None, score_finished=True
            )
