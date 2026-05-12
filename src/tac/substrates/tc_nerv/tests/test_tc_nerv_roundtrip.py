"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for tc_nerv.

Mirrors ``sane_hnerv/tests/test_sane_hnerv_roundtrip.py`` adapted for the
TCV1 grammar (extra tc_table section).
"""

from __future__ import annotations

import torch

from tac.substrates.tc_nerv.archive import (
    TCV1_HEADER_SIZE,
    TCV1_MAGIC,
    TCV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from tac.substrates.tc_nerv.architecture import TCNervConfig, TCNervSubstrate
from tac.substrates.tc_nerv.score_aware_loss import (
    TCNervScoreAwareLoss,
    TCScoreAwareLossWeights,
)


def _smoke_cfg() -> TCNervConfig:
    """Tiny config so tests run fast on CPU."""
    return TCNervConfig(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8, 6, 4, 4, 4),
        sin_frequency=30.0,
        num_pairs=4,
        output_height=24,
        output_width=32,
        num_upsample_blocks=3,
    )


def _meta_from_cfg(cfg: TCNervConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
    }


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = TCNervSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()
    tc_table = torch.tensor([1, -2, 3, -4], dtype=torch.int8)
    meta = _meta_from_cfg(cfg)

    blob = pack_archive(decoder_sd, latents, tc_table, meta)
    arc = parse_archive(blob)

    assert arc.schema_version == TCV1_SCHEMA_VERSION
    assert blob[:4] == TCV1_MAGIC

    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())
    for k, v in decoder_sd.items():
        rec = arc.decoder_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    assert arc.latents.shape == latents.shape
    quant_range = max(float(latents.max() - latents.min()), 1e-12)
    step = quant_range / 65534.0
    assert torch.allclose(arc.latents, latents, atol=step * 2.0)

    # tc_table preserved exactly (int8 -> int8)
    assert arc.tc_table.dtype == torch.int8
    assert arc.tc_table.shape == tc_table.shape
    assert torch.equal(arc.tc_table, tc_table)


def test_header_size_invariant_is_25_bytes():
    assert TCV1_HEADER_SIZE == 25


def test_parse_archive_rejects_short_blob():
    try:
        parse_archive(b"\x00")
    except ValueError as exc:
        assert "too short" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on short blob")


def test_parse_archive_rejects_wrong_magic():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = TCNervSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    tc_table = torch.zeros(cfg.num_pairs, dtype=torch.int8)
    meta = _meta_from_cfg(cfg)
    blob = bytearray(pack_archive(decoder_sd, latents, tc_table, meta))
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_pack_rejects_tc_table_wrong_dtype():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = TCNervSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    bad_tc = torch.zeros(cfg.num_pairs, dtype=torch.float32)
    meta = _meta_from_cfg(cfg)
    try:
        pack_archive(decoder_sd, latents, bad_tc, meta)
    except ValueError as exc:
        assert "int8" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on non-int8 tc_table")


def test_pack_rejects_tc_table_wrong_length():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = TCNervSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    bad_tc = torch.zeros(cfg.num_pairs + 3, dtype=torch.int8)
    meta = _meta_from_cfg(cfg)
    try:
        pack_archive(decoder_sd, latents, bad_tc, meta)
    except ValueError as exc:
        assert "tc_table length" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on length mismatch")


def test_forward_pass_after_roundtrip_matches_original_within_fp16_tolerance():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = TCNervSubstrate(cfg).eval()

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()
    tc_table = torch.zeros(cfg.num_pairs, dtype=torch.int8)
    meta = _meta_from_cfg(cfg)
    blob = pack_archive(decoder_sd, latents, tc_table, meta)
    arc = parse_archive(blob)

    rebuilt = TCNervSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.latents.copy_(arc.latents.to(rebuilt.latents.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


# Catalog #139 byte-mutation no_op_proof
def test_byte_mutation_changes_inflate_output_no_op_proof():
    """Mutating a latent or tc_table byte must produce different archive bytes
    AND a different parsed archive.
    """
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = TCNervSubstrate(cfg).eval()
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    tc_table = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
    meta = _meta_from_cfg(cfg)

    blob_a = pack_archive(decoder_sd, latents, tc_table, meta)

    # Mutate one latent
    mutated = latents.clone()
    mutated[0, 0] = mutated[0, 0] + 1.0
    blob_b = pack_archive(decoder_sd, mutated, tc_table, meta)
    assert blob_a != blob_b, "no_op_proof: latent mutation must change bytes"

    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)

    # Mutate the tc_table
    mutated_tc = tc_table.clone()
    mutated_tc[0] = 99
    blob_c = pack_archive(decoder_sd, latents, mutated_tc, meta)
    assert blob_a != blob_c, "no_op_proof: tc_table mutation must change bytes"
    arc_c = parse_archive(blob_c)
    assert int(arc_c.tc_table[0]) == 99


def test_score_aware_loss_temporal_consistency_term_is_in_parts():
    """L0 SKETCH smoke for the Lagrangian: the tc_term IS reported in parts.

    Uses identity scorers so the test is fast on CPU; the full
    SegNet/PoseNet integration is deferred to L1.
    """
    torch.manual_seed(31)

    # Identity stubs that satisfy the scorer signatures used by forward()
    class _SegStub(torch.nn.Module):
        """Identity-ish stub: takes (B, T=1, C, H, W) and projects to (B, 5, h, w)
        as a linear function of input so grad flows.
        """

        def forward(self, x):
            x = x[:, 0]  # (B, C, H, W)
            pad = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])
            return torch.cat([x, pad], dim=1)

    class _PoseStub(torch.nn.Module):
        """Identity-ish stub: per-channel mean of input concatenated to zeros."""

        def forward(self, x):
            m = x.mean(dim=(-2, -1))  # (B, 6)
            pad = torch.zeros(m.shape[0], 6)
            return torch.cat([m, pad], dim=1)

    seg = _SegStub()
    pose = _PoseStub()
    weights = TCScoreAwareLossWeights(lambda_tc=0.5)
    loss_fn = TCNervScoreAwareLoss(seg, pose, weights)

    rgb_0 = torch.rand(2, 3, 12, 16, requires_grad=True)
    rgb_1 = torch.rand(2, 3, 12, 16, requires_grad=True)
    gt_0 = torch.rand(2, 3, 12, 16)
    gt_1 = torch.rand(2, 3, 12, 16)
    archive_bytes = torch.tensor(100_000.0)

    loss, parts = loss_fn(rgb_0, rgb_1, gt_0, gt_1, archive_bytes)
    assert "tc_term" in parts
    assert parts["tc_term"].item() >= 0
    # The loss must be finite and differentiable
    assert torch.isfinite(loss)
    loss.backward()
    assert rgb_0.grad is not None
    assert rgb_1.grad is not None
