# SPDX-License-Identifier: MIT
"""PSV3 section-neutralization tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import torch

from tac.substrates.pact_nerv_selector_v3.architecture import (
    PactNervSelectorV3Config,
    PactNervSelectorV3Substrate,
)
from tac.substrates.pact_nerv_selector_v3.archive import pack_archive, parse_archive
from tac.substrates.pact_nerv_selector_v3.section_value import (
    neutralize_psv3_section,
    psv3_section_layout,
    write_zip_replacing_member,
)


def _cfg() -> PactNervSelectorV3Config:
    return PactNervSelectorV3Config(
        latent_dim=4,
        embed_dim=12,
        initial_grid_h=2,
        initial_grid_w=2,
        decoder_channels=(10, 8),
        num_upsample_blocks=2,
        num_pairs=2,
        output_height=8,
        output_width=8,
        selector_palette_size=16,
    )


def _meta(cfg: PactNervSelectorV3Config) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "rice_golomb_k": cfg.rice_golomb_k,
        "selector_palette_size": cfg.selector_palette_size,
    }


def _blob() -> bytes:
    torch.manual_seed(123)
    cfg = _cfg()
    model = PactNervSelectorV3Substrate(cfg)
    state = model.state_dict()
    return pack_archive(
        {k: v for k, v in state.items() if k not in {"latents", "selectors"}},
        state["latents"].clone(),
        b"\x00\x01\x02\x03",
        _meta(cfg),
        palette_size=cfg.selector_palette_size,
    )


def test_psv3_section_layout_sums_to_blob_size() -> None:
    blob = _blob()
    rows = psv3_section_layout(blob)
    assert [row.name for row in rows] == [
        "decoder_qw",
        "latents_rc",
        "selectors_rc",
        "receiver_state",
    ]
    assert rows[-1].offset + rows[-1].length == len(blob)
    assert all(len(row.sha256) == 64 for row in rows)


def test_neutralize_decoder_and_latents_stays_parseable() -> None:
    blob = _blob()
    dec = parse_archive(neutralize_psv3_section(blob, "decoder_qw"))
    assert all(float(t.abs().max()) == 0.0 for t in dec.decoder_state_dict.values())
    lat = parse_archive(neutralize_psv3_section(blob, "latents_rc"))
    assert float(lat.latents.abs().max()) == 0.0


def test_neutralize_selectors_removes_unused_selector_stream() -> None:
    arc = parse_archive(neutralize_psv3_section(_blob(), "selectors_rc"))
    assert arc.selector_bytes == b""


def test_write_zip_replacing_member_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("0.bin", b"old")
        zf.writestr("inflate.sh", b"#!/bin/sh\n")
    out_a = tmp_path / "a.zip"
    out_b = tmp_path / "b.zip"
    report_a = write_zip_replacing_member(
        source_archive=source,
        output_archive=out_a,
        member_name="0.bin",
        replacement_bytes=b"new",
    )
    report_b = write_zip_replacing_member(
        source_archive=source,
        output_archive=out_b,
        member_name="0.bin",
        replacement_bytes=b"new",
    )
    assert report_a["output_archive"]["sha256"] == report_b["output_archive"]["sha256"]
    with zipfile.ZipFile(out_a) as zf:
        assert zf.read("0.bin") == b"new"
        assert zf.read("inflate.sh") == b"#!/bin/sh\n"
