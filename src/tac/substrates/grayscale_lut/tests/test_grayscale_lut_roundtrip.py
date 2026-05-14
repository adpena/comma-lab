# SPDX-License-Identifier: MIT
"""Catalog #91 + #139 roundtrip tests for the grayscale_lut substrate."""

from __future__ import annotations

import torch

from tac.substrates.grayscale_lut.architecture import (
    GrayscaleLutConfig,
    GrayscaleLutSubstrate,
)
from tac.substrates.grayscale_lut.archive import (
    GLV1_HEADER_SIZE,
    GLV1_MAGIC,
    GLV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> GrayscaleLutConfig:
    return GrayscaleLutConfig(
        grayscale_downsample=4,
        decoder_hidden=8,
        decoder_blocks=2,
        embedding_dim=4,
        num_pairs=3,
        output_height=16,
        output_width=24,
    )


def _meta(cfg: GrayscaleLutConfig) -> dict[str, object]:
    return {
        "decoder_hidden": cfg.decoder_hidden,
        "decoder_blocks": cfg.decoder_blocks,
    }


def _build_smoke_inputs():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = GrayscaleLutSubstrate(cfg)
    sd = model.runtime_state_dict_for_archive()
    grayscale = model.quantize_grayscale_for_archive()
    return cfg, model, sd, grayscale, _meta(cfg)


def test_archive_pack_then_parse_recovers_components() -> None:
    cfg, _, sd, grayscale, meta = _build_smoke_inputs()
    blob = pack_archive(
        sd,
        grayscale,
        meta,
        num_pairs=cfg.num_pairs,
        grayscale_downsample=cfg.grayscale_downsample,
        embedding_dim=cfg.embedding_dim,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )
    arc = parse_archive(blob)

    assert blob[:4] == GLV1_MAGIC
    assert arc.schema_version == GLV1_SCHEMA_VERSION
    assert arc.num_pairs == cfg.num_pairs
    assert arc.grayscale_downsample == cfg.grayscale_downsample
    assert arc.embedding_dim == cfg.embedding_dim
    assert torch.equal(arc.grayscale, grayscale)
    assert set(arc.decoder_state_dict) == set(sd)
    for key, tensor in sd.items():
        restored = arc.decoder_state_dict[key]
        assert restored.shape == tensor.shape, key
        assert torch.allclose(restored.float(), tensor.float(), atol=1e-2), key


def test_header_size_invariant_is_30_bytes() -> None:
    assert GLV1_HEADER_SIZE == 30


def test_runtime_state_dict_excludes_grayscale_section() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(17)
    model = GrayscaleLutSubstrate(cfg)
    runtime_sd = model.runtime_state_dict_for_archive()

    assert "grayscale" not in runtime_sd
    assert "pair_embedding" in runtime_sd
    assert any(k.startswith("stem.") for k in runtime_sd)
    assert any(k.startswith("blocks.") for k in runtime_sd)
    assert any(k.startswith("head_rgb_0.") for k in runtime_sd)
    assert any(k.startswith("head_rgb_1.") for k in runtime_sd)


def test_pack_archive_rejects_grayscale_in_decoder_state_dict() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(19)
    model = GrayscaleLutSubstrate(cfg)
    grayscale = model.quantize_grayscale_for_archive()

    try:
        pack_archive(
            model.state_dict(),
            grayscale,
            _meta(cfg),
            num_pairs=cfg.num_pairs,
            grayscale_downsample=cfg.grayscale_downsample,
            embedding_dim=cfg.embedding_dim,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
        )
    except ValueError as exc:
        assert "contains grayscale" in str(exc)
        assert "runtime_state_dict_for_archive" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on duplicated grayscale section")


def test_parse_archive_rejects_wrong_magic() -> None:
    cfg, _, sd, grayscale, meta = _build_smoke_inputs()
    blob = bytearray(
        pack_archive(
            sd,
            grayscale,
            meta,
            num_pairs=cfg.num_pairs,
            grayscale_downsample=cfg.grayscale_downsample,
            embedding_dim=cfg.embedding_dim,
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


def test_byte_mutation_changes_grayscale_payload() -> None:
    cfg, _, sd, grayscale, meta = _build_smoke_inputs()
    blob_a = pack_archive(
        sd,
        grayscale,
        meta,
        num_pairs=cfg.num_pairs,
        grayscale_downsample=cfg.grayscale_downsample,
        embedding_dim=cfg.embedding_dim,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )
    grayscale_b = grayscale.clone()
    grayscale_b[0, 0, 0, 0] = (int(grayscale_b[0, 0, 0, 0]) + 1) % 256
    blob_b = pack_archive(
        sd,
        grayscale_b,
        meta,
        num_pairs=cfg.num_pairs,
        grayscale_downsample=cfg.grayscale_downsample,
        embedding_dim=cfg.embedding_dim,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
    )

    assert blob_a != blob_b
    assert not torch.equal(parse_archive(blob_a).grayscale, parse_archive(blob_b).grayscale)


def test_substrate_forward_shape() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(23)
    model = GrayscaleLutSubstrate(cfg).eval()
    with torch.no_grad():
        rgb_0, rgb_1 = model(torch.tensor([0, 1], dtype=torch.long))

    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert torch.all((rgb_0 >= 0.0) & (rgb_0 <= 1.0))
    assert torch.all((rgb_1 >= 0.0) & (rgb_1 <= 1.0))

