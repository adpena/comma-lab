# SPDX-License-Identifier: MIT
"""PSV4 section-neutralization tests."""

from __future__ import annotations

import pytest
import torch

from tac.substrates.pact_nerv_selector_v4.architecture import (
    PactNervSelectorV4Config,
    PactNervSelectorV4Substrate,
)
from tac.substrates.pact_nerv_selector_v4.archive import pack_archive, parse_archive
from tac.substrates.pact_nerv_selector_v4.section_value import (
    neutralize_psv4_section,
    psv4_layout_report,
    psv4_section_layout,
)


def _cfg() -> PactNervSelectorV4Config:
    return PactNervSelectorV4Config(
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


def _meta(cfg: PactNervSelectorV4Config) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "selector_palette_size": cfg.selector_palette_size,
    }


def _blob() -> bytes:
    torch.manual_seed(123)
    cfg = _cfg()
    model = PactNervSelectorV4Substrate(cfg)
    state = model.state_dict()
    return pack_archive(
        {k: v for k, v in state.items() if k not in {"latents", "selectors"}},
        state["latents"].clone(),
        b"\x00\x01\x02\x03",
        _meta(cfg),
        palette_size=cfg.selector_palette_size,
    )


def test_psv4_section_layout_sums_to_blob_size() -> None:
    blob = _blob()
    rows = psv4_section_layout(blob)
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
    dec = parse_archive(neutralize_psv4_section(blob, "decoder_qw"))
    assert all(float(t.abs().max()) == 0.0 for t in dec.decoder_state_dict.values())
    lat = parse_archive(neutralize_psv4_section(blob, "latents_rc"))
    assert float(lat.latents.abs().max()) == 0.0


def test_neutralize_selectors_removes_charged_selector_stream() -> None:
    arc = parse_archive(neutralize_psv4_section(_blob(), "selectors_rc"))
    assert arc.selector_bytes == b""


def test_neutralize_receiver_state_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="receiver_state neutralization"):
        neutralize_psv4_section(_blob(), "receiver_state")


def test_psv4_layout_report_is_false_authority() -> None:
    report = psv4_layout_report(blob=_blob())
    assert report["schema"] == "pact_nerv_selector_v4_section_layout.v1"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert [row["name"] for row in report["sections"]] == [
        "decoder_qw",
        "latents_rc",
        "selectors_rc",
        "receiver_state",
    ]
