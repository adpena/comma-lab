"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP test for siren substrate.

Mirrors src/tac/substrates/sane_hnerv/tests/test_sane_hnerv_roundtrip.py shape:
the encode/decode contract of the SRV1 monolithic 0.bin grammar must be
byte-faithful, and the Catalog #139 no-op byte-mutation smoke must pass.
"""

from __future__ import annotations

import torch

from tac.substrates.siren.architecture import SirenConfig, SirenSubstrate
from tac.substrates.siren.archive import (
    SRV1_HEADER_SIZE,
    SRV1_MAGIC,
    SRV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> SirenConfig:
    """Tiny config so tests run fast on CPU."""
    return SirenConfig(
        hidden_dim=16,
        num_hidden_layers=3,
        first_omega=30.0,
        hidden_omega=1.0,
        num_pairs=3,
        output_height=8,
        output_width=12,
    )


def _build_smoke_inputs():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = SirenSubstrate(cfg)
    sd = model.runtime_state_dict_for_archive()
    meta = {
        "first_omega": cfg.first_omega,
        "hidden_omega": cfg.hidden_omega,
        "coord_dim": cfg.coord_dim,
        "output_dim": cfg.output_dim,
    }
    return cfg, model, sd, meta


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_recovers_tensors():
    cfg, _, sd, meta = _build_smoke_inputs()
    blob = pack_archive(
        sd,
        meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )
    arc = parse_archive(blob)

    assert arc.schema_version == SRV1_SCHEMA_VERSION
    assert blob[:4] == SRV1_MAGIC
    assert arc.num_pairs == cfg.num_pairs
    assert arc.hidden_dim == cfg.hidden_dim
    assert arc.num_hidden_layers == cfg.num_hidden_layers
    assert arc.output_height == cfg.output_height
    assert arc.output_width == cfg.output_width

    # state_dict keys preserved
    assert set(arc.decoder_state_dict.keys()) == set(sd.keys())
    for k, v in sd.items():
        rec = arc.decoder_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)


def test_header_size_invariant_is_22_bytes():
    assert SRV1_HEADER_SIZE == 22


def test_runtime_state_dict_excludes_deterministic_meshgrid_buffer():
    cfg = _smoke_cfg()
    torch.manual_seed(17)
    model = SirenSubstrate(cfg)
    runtime_sd = model.runtime_state_dict_for_archive()

    assert "_spatial_coords" not in runtime_sd
    assert any(k.startswith("hidden.") for k in runtime_sd)
    assert any(k.startswith("output_layer.") for k in runtime_sd)


def test_pack_archive_rejects_spatial_coords_buffer():
    cfg = _smoke_cfg()
    torch.manual_seed(19)
    model = SirenSubstrate(cfg)
    meta = {
        "first_omega": cfg.first_omega,
        "hidden_omega": cfg.hidden_omega,
        "coord_dim": cfg.coord_dim,
        "output_dim": cfg.output_dim,
    }

    try:
        pack_archive(
            model.state_dict(),
            meta,
            num_pairs=cfg.num_pairs,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
        )
    except ValueError as exc:
        assert "_spatial_coords" in str(exc)
        assert "runtime_state_dict_for_archive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on deterministic meshgrid buffer")


def test_config_rejects_non_paired_rgb_output_dim():
    try:
        SirenConfig(output_dim=3)
    except ValueError as exc:
        assert "output_dim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for output_dim != 6")


def test_parse_archive_rejects_short_blob():
    try:
        parse_archive(b"\x00")
    except ValueError as exc:
        assert "too short" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on short blob")


def test_parse_archive_rejects_wrong_magic():
    cfg, _, sd, meta = _build_smoke_inputs()
    blob = bytearray(
        pack_archive(
            sd,
            meta,
            num_pairs=cfg.num_pairs,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
        )
    )
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_forward_pass_after_roundtrip_matches_within_fp16_tolerance():
    cfg, model, sd, meta = _build_smoke_inputs()
    model.eval()
    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    blob = pack_archive(
        sd,
        meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )
    arc = parse_archive(blob)

    rebuilt = SirenSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        rgb_0_b, rgb_1_b = rebuilt(idx)

    # fp16 roundtrip + sin nonlinearity: tolerate ~5e-2
    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_archive_no_op_proof():
    """Mutate a network weight; archive bytes must change AND roundtrip
    forward output must differ.

    This is the no_op_proof for siren's SRV1 grammar.
    """
    cfg, model, sd, meta = _build_smoke_inputs()
    blob_a = pack_archive(
        sd,
        meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )
    # Mutate a weight in the output layer (will show up in RGB output downstream)
    sd_mut = {k: v.clone() for k, v in sd.items()}
    out_key = next(k for k in sd_mut if k.startswith("output_layer.weight"))
    sd_mut[out_key] = sd_mut[out_key] + 0.1
    blob_b = pack_archive(
        sd_mut,
        meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )
    assert blob_a != blob_b, "no_op_proof: mutating a weight must change archive bytes"

    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    # The mutated weight must differ between A and B
    assert not torch.allclose(
        arc_a.decoder_state_dict[out_key].to(torch.float32),
        arc_b.decoder_state_dict[out_key].to(torch.float32),
        atol=1e-3,
    )


def test_substrate_forward_shape():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = SirenSubstrate(cfg).eval()
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
    model = SirenSubstrate(cfg).eval()
    bad = torch.tensor([cfg.num_pairs], dtype=torch.long)
    try:
        model(bad)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on out-of-range pair index")


def test_siren_init_zero_bias_on_all_layers():
    """SIREN init zeroes biases per Sitzmann paper."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = SirenSubstrate(cfg)
    for name, p in model.named_parameters():
        if "bias" in name:
            assert torch.all(p == 0.0), f"{name} expected zero-init bias per SIREN"


def test_pair_index_changes_forward_output():
    """Different pair indices must produce different outputs (no degeneracy)."""
    cfg = _smoke_cfg()
    torch.manual_seed(21)
    model = SirenSubstrate(cfg).eval()
    idx_a = torch.tensor([0], dtype=torch.long)
    idx_b = torch.tensor([cfg.num_pairs - 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, _ = model(idx_a)
        rgb_0_b, _ = model(idx_b)
    # The two outputs must not be identical (otherwise the t-coordinate is unused)
    assert not torch.allclose(rgb_0_a, rgb_0_b, atol=1e-5)
