"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP test for vq_vae substrate.

Mirrors src/tac/substrates/sane_hnerv/tests/test_sane_hnerv_roundtrip.py shape:
the encode/decode contract of the VQV1 monolithic 0.bin grammar must be
byte-faithful, and the Catalog #139 no-op byte-mutation smoke must pass.
"""

from __future__ import annotations

import torch

from tac.substrates.vq_vae.architecture import VqVaeConfig, VqVaeSubstrate
from tac.substrates.vq_vae.archive import (
    VQV1_HEADER_SIZE,
    VQV1_MAGIC,
    VQV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> VqVaeConfig:
    """Tiny config so tests run fast on CPU."""
    return VqVaeConfig(
        codebook_size=16,
        embedding_dim=4,
        encoder_hidden=8,
        decoder_hidden=8,
        grid_downsample=8,
        num_pairs=3,
        output_height=16,
        output_width=24,
    )


def _build_smoke_inputs():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = VqVaeSubstrate(cfg)
    sd = model.runtime_state_dict_for_archive()
    indices = model.encode_indices_for_archive()  # (num_pairs, 2, h_grid, w_grid)
    meta = {
        "encoder_hidden": cfg.encoder_hidden,
        "decoder_hidden": cfg.decoder_hidden,
        "grid_downsample": cfg.grid_downsample,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }
    return cfg, model, sd, indices, meta


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_recovers_tensors():
    cfg, _, sd, indices, meta = _build_smoke_inputs()
    blob = pack_archive(
        sd,
        indices,
        meta,
        codebook_size=cfg.codebook_size,
        embedding_dim=cfg.embedding_dim,
    )
    arc = parse_archive(blob)

    assert arc.schema_version == VQV1_SCHEMA_VERSION
    assert blob[:4] == VQV1_MAGIC
    assert arc.codebook_size == cfg.codebook_size
    assert arc.embedding_dim == cfg.embedding_dim

    # state_dict keys preserved (fp16 cast may produce tiny error)
    assert set(arc.decoder_state_dict.keys()) == set(sd.keys())
    for k, v in sd.items():
        rec = arc.decoder_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    # Indices must roundtrip EXACTLY (they are int; no quantization step)
    assert arc.indices.shape == indices.shape
    assert torch.equal(arc.indices, indices)


def test_header_size_invariant_is_27_bytes():
    assert VQV1_HEADER_SIZE == 27


def test_runtime_state_dict_excludes_training_only_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(17)
    model = VqVaeSubstrate(cfg)
    runtime_sd = model.runtime_state_dict_for_archive()

    assert "codebook" in runtime_sd
    assert any(k.startswith("decoder.") for k in runtime_sd)
    assert "per_pair_features" not in runtime_sd
    assert not any(k.startswith("encoder_refine.") for k in runtime_sd)


def test_pack_archive_rejects_training_only_state_dict():
    cfg = _smoke_cfg()
    torch.manual_seed(19)
    model = VqVaeSubstrate(cfg)
    indices = model.encode_indices_for_archive()
    meta = {
        "encoder_hidden": cfg.encoder_hidden,
        "decoder_hidden": cfg.decoder_hidden,
        "grid_downsample": cfg.grid_downsample,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }

    try:
        pack_archive(
            model.state_dict(),
            indices,
            meta,
            codebook_size=cfg.codebook_size,
            embedding_dim=cfg.embedding_dim,
        )
    except ValueError as exc:
        assert "training-only" in str(exc)
        assert "runtime_state_dict_for_archive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on training-only state_dict")


def test_config_rejects_non_power_of_two_grid_downsample():
    try:
        VqVaeConfig(grid_downsample=6)
    except ValueError as exc:
        assert "power of two" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-power-of-two grid_downsample")


def test_parse_archive_rejects_short_blob():
    try:
        parse_archive(b"\x00")
    except ValueError as exc:
        assert "too short" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on short blob")


def test_parse_archive_rejects_wrong_magic():
    cfg, _, sd, indices, meta = _build_smoke_inputs()
    blob = bytearray(
        pack_archive(
            sd,
            indices,
            meta,
            codebook_size=cfg.codebook_size,
            embedding_dim=cfg.embedding_dim,
        )
    )
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_archive_no_op_proof():
    """Mutate an index entry; archive bytes must change AND roundtrip indices must differ.

    This is the no_op_proof for vq_vae's VQV1 grammar.
    """
    cfg, _, sd, indices, meta = _build_smoke_inputs()
    blob_a = pack_archive(
        sd,
        indices,
        meta,
        codebook_size=cfg.codebook_size,
        embedding_dim=cfg.embedding_dim,
    )
    indices_mut = indices.clone()
    # Pick a position whose change is meaningful: flip to a different codebook entry
    original = int(indices_mut[0, 0, 0, 0])
    indices_mut[0, 0, 0, 0] = (original + 1) % cfg.codebook_size
    blob_b = pack_archive(
        sd,
        indices_mut,
        meta,
        codebook_size=cfg.codebook_size,
        embedding_dim=cfg.embedding_dim,
    )
    assert blob_a != blob_b, "no_op_proof: mutating an index must change archive bytes"

    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert int(arc_a.indices[0, 0, 0, 0]) != int(arc_b.indices[0, 0, 0, 0])


def test_substrate_forward_shape():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = VqVaeSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert torch.all(rgb_0 >= 0.0) and torch.all(rgb_0 <= 1.0)
    assert torch.all(rgb_1 >= 0.0) and torch.all(rgb_1 <= 1.0)


def test_commitment_loss_is_finite_scalar():
    cfg = _smoke_cfg()
    torch.manual_seed(11)
    model = VqVaeSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    loss = model.compute_commitment_loss(idx)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_index_packing_full_range_roundtrip():
    """The int16-offset packing must support the full [0, K) range for K up to 65536."""
    from tac.substrates.vq_vae.archive import _pack_indices_int16, _unpack_indices_int16

    cfg = _smoke_cfg()
    # Construct indices that cover near-min and near-max codebook entries
    indices = torch.tensor(
        [[0, cfg.codebook_size - 1], [1, cfg.codebook_size // 2]],
        dtype=torch.int64,
    ).view(1, 2, 1, 2)
    blob = _pack_indices_int16(indices, cfg.codebook_size)
    out = _unpack_indices_int16(blob, indices.shape)
    assert torch.equal(out, indices)


def test_pair_indices_out_of_range_raises():
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = VqVaeSubstrate(cfg).eval()
    bad = torch.tensor([cfg.num_pairs], dtype=torch.long)
    try:
        model(bad)
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on out-of-range pair index")
