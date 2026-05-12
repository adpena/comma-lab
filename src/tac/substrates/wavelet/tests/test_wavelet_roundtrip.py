"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP test for wavelet substrate.

Mirrors src/tac/substrates/sane_hnerv/tests/test_sane_hnerv_roundtrip.py shape:
encode/decode contract of the WLV1 monolithic 0.bin grammar must be
byte-faithful, and the Catalog #139 no-op byte-mutation smoke must pass.
"""

from __future__ import annotations

import torch

from tac.substrates.wavelet.archive import (
    WLV1_HEADER_SIZE,
    WLV1_MAGIC,
    WLV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from tac.substrates.wavelet.architecture import WaveletConfig, WaveletSubstrate


def _smoke_cfg() -> WaveletConfig:
    """Tiny config so tests run fast on CPU."""
    return WaveletConfig(
        coeff_channels=2,
        synthesis_hidden=8,
        synthesis_layers=2,
        num_pairs=3,
        output_height=16,
        output_width=24,
    )


def _build_smoke_inputs():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = WaveletSubstrate(cfg)
    synth_sd = model.synthesis.state_dict()
    film_sd = {"film": model.film.clone()}
    LL = model.coeff_ll.clone()
    LH = model.coeff_lh.clone()
    HL = model.coeff_hl.clone()
    HH = model.coeff_hh.clone()
    meta = {
        "synthesis_hidden": cfg.synthesis_hidden,
        "synthesis_layers": cfg.synthesis_layers,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }
    return cfg, model, synth_sd, film_sd, LL, LH, HL, HH, meta


def test_archive_pack_then_parse_recovers_tensors():
    _, _, synth_sd, film_sd, LL, LH, HL, HH, meta = _build_smoke_inputs()
    blob = pack_archive(synth_sd, film_sd, LL, LH, HL, HH, meta)
    arc = parse_archive(blob)
    assert arc.schema_version == WLV1_SCHEMA_VERSION
    assert blob[:4] == WLV1_MAGIC

    assert set(arc.synthesis_state_dict.keys()) == set(synth_sd.keys())
    assert set(arc.film_state_dict.keys()) == set(film_sd.keys())

    for nm, (rec, orig) in (
        ("LL", (arc.LL, LL)),
        ("LH", (arc.LH, LH)),
        ("HL", (arc.HL, HL)),
        ("HH", (arc.HH, HH)),
    ):
        assert rec.shape == orig.shape, f"{nm} shape changed"
        rng = max(float(orig.max() - orig.min()), 1e-12)
        assert torch.allclose(rec, orig, atol=(rng / 65534.0) * 2.0), f"{nm} dequant mismatch"


def test_header_size_invariant_is_41_bytes():
    assert WLV1_HEADER_SIZE == 41


def test_parse_archive_rejects_short_blob():
    try:
        parse_archive(b"\x00")
    except ValueError as exc:
        assert "too short" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on short blob")


def test_parse_archive_rejects_wrong_magic():
    _, _, synth_sd, film_sd, LL, LH, HL, HH, meta = _build_smoke_inputs()
    blob = bytearray(pack_archive(synth_sd, film_sd, LL, LH, HL, HH, meta))
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_forward_pass_after_roundtrip_matches_within_fp16_tolerance():
    cfg, model, synth_sd, film_sd, LL, LH, HL, HH, meta = _build_smoke_inputs()
    model.eval()
    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    blob = pack_archive(synth_sd, film_sd, LL, LH, HL, HH, meta)
    arc = parse_archive(blob)

    rebuilt = WaveletSubstrate(cfg).eval()
    rebuilt.synthesis.load_state_dict(arc.synthesis_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.film.copy_(arc.film_state_dict["film"].to(rebuilt.film.dtype))
        rebuilt.coeff_ll.copy_(arc.LL.to(rebuilt.coeff_ll.dtype))
        rebuilt.coeff_lh.copy_(arc.LH.to(rebuilt.coeff_lh.dtype))
        rebuilt.coeff_hl.copy_(arc.HL.to(rebuilt.coeff_hl.dtype))
        rebuilt.coeff_hh.copy_(arc.HH.to(rebuilt.coeff_hh.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


def test_byte_mutation_changes_archive_no_op_proof():
    """Catalog #139 no_op_proof: mutate an LL subband element; archive bytes
    must change AND parsed subband must reflect the change after roundtrip.
    """
    _, _, synth_sd, film_sd, LL, LH, HL, HH, meta = _build_smoke_inputs()
    blob_a = pack_archive(synth_sd, film_sd, LL, LH, HL, HH, meta)

    LL_mut = LL.clone()
    LL_mut[0, 0, 0, 0] = LL_mut[0, 0, 0, 0] + 1.0
    blob_b = pack_archive(synth_sd, film_sd, LL_mut, LH, HL, HH, meta)
    assert blob_a != blob_b, "no_op_proof: mutating LL subband must change archive bytes"

    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.LL[0, 0, 0, 0], arc_b.LL[0, 0, 0, 0], atol=1e-6)


def test_substrate_forward_shape():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = WaveletSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert torch.all(rgb_0 >= 0.0) and torch.all(rgb_0 <= 1.0)
    assert torch.all(rgb_1 >= 0.0) and torch.all(rgb_1 <= 1.0)


def test_pair_indices_out_of_range_raises():
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = WaveletSubstrate(cfg).eval()
    bad = torch.tensor([cfg.num_pairs], dtype=torch.long)
    try:
        model(bad)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on out-of-range pair index")


def test_db4_filter_coefficients_unit_norm():
    """Daubechies-4 low-pass filter coefficients should be orthonormal."""
    from tac.substrates.wavelet.architecture import _DB4_LO, _db4_hi

    lo = _DB4_LO
    hi = _db4_hi()

    # sum(lo^2) = 1 (orthonormality)
    sum_sq_lo = sum(x * x for x in lo)
    assert abs(sum_sq_lo - 1.0) < 1e-6

    # sum(hi^2) = 1
    sum_sq_hi = sum(x * x for x in hi)
    assert abs(sum_sq_hi - 1.0) < 1e-6
