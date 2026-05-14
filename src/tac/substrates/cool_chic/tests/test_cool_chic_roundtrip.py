# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP test for cool_chic substrate.

Mirrors src/tac/substrates/sane_hnerv/tests/test_sane_hnerv_roundtrip.py shape:
the encode/decode contract of the CCV1 monolithic 0.bin grammar must be
byte-faithful, and the Catalog #139 no-op byte-mutation smoke must pass.
"""

from __future__ import annotations

import torch

from tac.substrates.cool_chic.archive import (
    CCV1_HEADER_SIZE,
    CCV1_MAGIC,
    CCV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from tac.substrates.cool_chic.architecture import CoolChicConfig, CoolChicSubstrate


def _smoke_cfg() -> CoolChicConfig:
    """Tiny config so tests run fast on CPU."""
    return CoolChicConfig(
        latent_channels_coarse=2,
        latent_channels_fine=2,
        coarse_scale_factor=8,
        fine_scale_factor=4,
        synthesis_hidden=8,
        synthesis_layers=2,
        ar_prior_hidden=4,
        num_pairs=3,
        output_height=16,
        output_width=24,
    )


def _build_smoke_inputs():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = CoolChicSubstrate(cfg)
    synth_sd = model.synthesis.state_dict()
    # Pack AR prior nets as prefixed keys
    ar_sd = {}
    for k, v in model.ar_prior_coarse.state_dict().items():
        ar_sd[f"coarse.{k}"] = v
    for k, v in model.ar_prior_fine.state_dict().items():
        ar_sd[f"fine.{k}"] = v
    latents_coarse = model.latents_coarse.clone()
    latents_fine = model.latents_fine.clone()
    meta = {
        "coarse_scale_factor": cfg.coarse_scale_factor,
        "fine_scale_factor": cfg.fine_scale_factor,
        "synthesis_hidden": cfg.synthesis_hidden,
        "synthesis_layers": cfg.synthesis_layers,
        "ar_prior_hidden": cfg.ar_prior_hidden,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }
    return cfg, model, synth_sd, ar_sd, latents_coarse, latents_fine, meta


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_recovers_tensors():
    _, _, synth_sd, ar_sd, lc, lf, meta = _build_smoke_inputs()
    blob = pack_archive(synth_sd, ar_sd, lc, lf, meta)
    arc = parse_archive(blob)
    assert arc.schema_version == CCV1_SCHEMA_VERSION
    assert blob[:4] == CCV1_MAGIC

    assert set(arc.synthesis_state_dict.keys()) == set(synth_sd.keys())
    for k, v in synth_sd.items():
        rec = arc.synthesis_state_dict[k]
        assert rec.shape == v.shape
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    assert set(arc.ar_prior_state_dict.keys()) == set(ar_sd.keys())

    assert arc.latents_coarse.shape == lc.shape
    assert arc.latents_fine.shape == lf.shape

    coarse_range = max(float(lc.max() - lc.min()), 1e-12)
    fine_range = max(float(lf.max() - lf.min()), 1e-12)
    assert torch.allclose(arc.latents_coarse, lc, atol=(coarse_range / 65534.0) * 2.0)
    assert torch.allclose(arc.latents_fine, lf, atol=(fine_range / 65534.0) * 2.0)


def test_header_size_invariant_is_39_bytes():
    assert CCV1_HEADER_SIZE == 39


def test_parse_archive_rejects_short_blob():
    try:
        parse_archive(b"\x00")
    except ValueError as exc:
        assert "too short" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on short blob")


def test_parse_archive_rejects_wrong_magic():
    _, _, synth_sd, ar_sd, lc, lf, meta = _build_smoke_inputs()
    blob = bytearray(pack_archive(synth_sd, ar_sd, lc, lf, meta))
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_forward_pass_after_roundtrip_matches_within_fp16_tolerance():
    cfg, model, synth_sd, ar_sd, lc, lf, meta = _build_smoke_inputs()
    model.eval()
    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    blob = pack_archive(synth_sd, ar_sd, lc, lf, meta)
    arc = parse_archive(blob)

    rebuilt = CoolChicSubstrate(cfg).eval()
    rebuilt.synthesis.load_state_dict(arc.synthesis_state_dict, strict=False)
    rebuilt.ar_prior_coarse.load_state_dict(
        {k.replace("coarse.", ""): v for k, v in arc.ar_prior_state_dict.items() if k.startswith("coarse.")},
        strict=False,
    )
    rebuilt.ar_prior_fine.load_state_dict(
        {k.replace("fine.", ""): v for k, v in arc.ar_prior_state_dict.items() if k.startswith("fine.")},
        strict=False,
    )
    with torch.no_grad():
        rebuilt.latents_coarse.copy_(arc.latents_coarse.to(rebuilt.latents_coarse.dtype))
        rebuilt.latents_fine.copy_(arc.latents_fine.to(rebuilt.latents_fine.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_archive_no_op_proof():
    """Mutate a latent; archive bytes must change AND roundtrip latents must
    differ. This is the no_op_proof for cool_chic's CCV1 grammar.
    """
    _, _, synth_sd, ar_sd, lc, lf, meta = _build_smoke_inputs()
    blob_a = pack_archive(synth_sd, ar_sd, lc, lf, meta)
    lc_mut = lc.clone()
    lc_mut[0, 0, 0, 0] = lc_mut[0, 0, 0, 0] + 1.0  # large delta beyond int16 quant step
    blob_b = pack_archive(synth_sd, ar_sd, lc_mut, lf, meta)
    assert blob_a != blob_b, "no_op_proof: mutating coarse latent must change archive bytes"

    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents_coarse[0, 0, 0, 0], arc_b.latents_coarse[0, 0, 0, 0], atol=1e-6)


def test_substrate_forward_shape():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = CoolChicSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert torch.all(rgb_0 >= 0.0) and torch.all(rgb_0 <= 1.0)
    assert torch.all(rgb_1 >= 0.0) and torch.all(rgb_1 <= 1.0)


def test_ar_log_prob_is_scalar_and_finite():
    cfg = _smoke_cfg()
    torch.manual_seed(11)
    model = CoolChicSubstrate(cfg).eval()
    log_p = model.compute_ar_log_prob()
    assert log_p.dim() == 0
    assert torch.isfinite(log_p)


def test_pair_indices_out_of_range_raises():
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = CoolChicSubstrate(cfg).eval()
    bad = torch.tensor([cfg.num_pairs], dtype=torch.long)
    try:
        model(bad)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on out-of-range pair index")
