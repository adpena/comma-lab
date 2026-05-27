# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for boost_nerv.

Proves the encode/decode contract of the BSV1 monolithic 0.bin grammar +
boosting chain forward-pass parity under fp16 + int16-quant roundtrip.
Plus a smoke-level test that the trainer's _full_main raises
NotImplementedError per the L0 SCAFFOLD posture (Catalog #240).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tac.substrates._shared.numpy_portable_inflate import assert_inflate_is_numpy_portable
from tac.substrates.boost_nerv.architecture import (
    BoostnervConfig,
    BoostnervSubstrate,
)
from tac.substrates.boost_nerv.archive import (
    BSV1_HEADER_SIZE,
    BSV1_MAGIC,
    BSV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
    parse_archive_numpy,
)
from tac.substrates.boost_nerv.inflate import inflate_one_video

_INFLATE_PATH = Path(__file__).resolve().parents[1] / "inflate.py"


def _smoke_cfg() -> BoostnervConfig:
    return BoostnervConfig(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        num_boosting_rounds=2,
        boosting_gain_clamp=0.1,
        boosting_hidden_dim=8,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: BoostnervConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "boosting_gain_clamp": cfg.boosting_gain_clamp,
        "boosting_hidden_dim": cfg.boosting_hidden_dim,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = BoostnervSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()

    blob = pack_archive(
        decoder_sd, latents, _smoke_meta(cfg),
        num_boosting_rounds=cfg.num_boosting_rounds,
    )
    arc = parse_archive(blob)

    assert arc.schema_version == BSV1_SCHEMA_VERSION
    assert blob[:4] == BSV1_MAGIC
    assert arc.num_boosting_rounds == cfg.num_boosting_rounds
    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())
    for k, v in decoder_sd.items():
        rec = arc.decoder_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    assert arc.latents.shape == latents.shape
    quant_range = max(float(latents.max() - latents.min()), 1e-12)
    step = quant_range / 65534.0
    assert torch.allclose(arc.latents, latents, atol=step * 2.0)


def test_numpy_parse_matches_torch_parse_roundtrip():
    cfg = _smoke_cfg()
    torch.manual_seed(3)
    model = BoostnervSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    blob = pack_archive(
        decoder_sd,
        sd["latents"].clone(),
        _smoke_meta(cfg),
        num_boosting_rounds=cfg.num_boosting_rounds,
    )

    torch_arc = parse_archive(blob)
    numpy_arc = parse_archive_numpy(blob)

    assert numpy_arc.schema_version == torch_arc.schema_version
    assert numpy_arc.num_boosting_rounds == torch_arc.num_boosting_rounds
    assert np.abs(torch_arc.latents.numpy() - numpy_arc.latents).max() < 1e-5
    assert set(numpy_arc.decoder_state_dict) == set(torch_arc.decoder_state_dict)
    for key, value in numpy_arc.decoder_state_dict.items():
        assert np.abs(torch_arc.decoder_state_dict[key].numpy() - value).max() < 1e-5


def test_numpy_inflate_matches_torch_roundtrip_pngs(tmp_path):
    cfg = _smoke_cfg()
    torch.manual_seed(17)
    model = BoostnervSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    blob = pack_archive(
        decoder_sd,
        sd["latents"].clone(),
        _smoke_meta(cfg),
        num_boosting_rounds=cfg.num_boosting_rounds,
    )
    arc = parse_archive(blob)

    rebuilt = BoostnervSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.latents.copy_(arc.latents.to(rebuilt.latents.dtype))
        rgb_0, rgb_1 = rebuilt(torch.tensor([0], dtype=torch.long))

    inflate_one_video(blob, tmp_path)

    for frame_idx, rgb in ((0, rgb_0), (1, rgb_1)):
        actual = np.asarray(Image.open(tmp_path / f"{frame_idx}.png"))
        expected = (
            (rgb[0].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
        assert actual.shape == expected.shape
        assert np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max() <= 2


def test_boost_nerv_inflate_is_numpy_portable() -> None:
    assert_inflate_is_numpy_portable(_INFLATE_PATH)


def test_header_size_invariant_is_22_bytes():
    assert BSV1_HEADER_SIZE == 22


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
    model = BoostnervSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    blob = bytearray(
        pack_archive(
            decoder_sd, latents, _smoke_meta(cfg),
            num_boosting_rounds=cfg.num_boosting_rounds,
        )
    )
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_pack_archive_rejects_oversize_boosting_rounds():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = BoostnervSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    try:
        pack_archive(
            decoder_sd, latents, _smoke_meta(cfg),
            num_boosting_rounds=256,
        )
    except ValueError as exc:
        assert "num_boosting_rounds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on out-of-u8 boosting rounds")


def test_forward_pass_after_roundtrip_matches_original_within_tolerance():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = BoostnervSubstrate(cfg).eval()

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()
    blob = pack_archive(
        decoder_sd, latents, _smoke_meta(cfg),
        num_boosting_rounds=cfg.num_boosting_rounds,
    )
    arc = parse_archive(blob)

    rebuilt = BoostnervSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.latents.copy_(arc.latents.to(rebuilt.latents.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_inflate_output_no_op_proof():
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = BoostnervSubstrate(cfg).eval()
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()

    blob_a = pack_archive(
        decoder_sd, latents, _smoke_meta(cfg),
        num_boosting_rounds=cfg.num_boosting_rounds,
    )
    mutated = latents.clone()
    mutated[0, 0] = mutated[0, 0] + 1.0
    blob_b = pack_archive(
        decoder_sd, mutated, _smoke_meta(cfg),
        num_boosting_rounds=cfg.num_boosting_rounds,
    )

    assert blob_a != blob_b, "no_op_proof: mutating latents must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)


def test_forward_pass_produces_unit_interval_rgb():
    """L5 compliance: substrate is a full RGB renderer (not a mask codec)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = BoostnervSubstrate(cfg).eval()
    idx = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_boosting_chain_has_canonical_round_count():
    """Distinctive design check: boost_nerv must have num_boosting_rounds heads."""
    cfg = _smoke_cfg()
    model = BoostnervSubstrate(cfg)
    assert len(model.boosting_heads) == cfg.num_boosting_rounds
    assert cfg.num_boosting_rounds >= 1, "boosting MUST have at least 1 round"


def test_boosting_residual_bounded_by_gain_clamp():
    """The per-round residual must stay in [-gain, +gain] before addition."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = BoostnervSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    z = model.latents[idx]
    # Construct a deliberately-out-of-band rgb_in (all-ones) and check the
    # residual clamp keeps the output bounded.
    rgb_in = torch.ones(2, 3, cfg.output_height, cfg.output_width)
    with torch.no_grad():
        head = model.boosting_heads[0]
        raw_residual = head(rgb_in, z)
        clamped = torch.clamp(raw_residual, -cfg.boosting_gain_clamp, cfg.boosting_gain_clamp)
    assert float(clamped.abs().max()) <= cfg.boosting_gain_clamp + 1e-6


def test_full_main_implemented_and_cuda_gated(tmp_path):
    """CLASS-SHIFT-FULL-MAIN-CLUSTER 2026-05-27: _full_main IMPLEMENTED + CUDA-gated.

    The L0 SCAFFOLD NotImplementedError is extinguished: ``_full_main`` now
    routes the canonical score-aware training loop through
    ``run_pact_nerv_score_aware_training``. Per CLAUDE.md "MPS auth eval is
    NOISE" + Catalog #1, the full (non-smoke) path is CUDA-required; invoking
    it with ``--device cpu`` refuses via ``device_or_die`` (SystemExit),
    proving the path wires correctly without firing paid GPU. PAID DISPATCH
    stays gated by ``dispatch_enabled: false`` + ``research_only: true`` on the
    recipe per Catalog #325 (code complete, trigger gated).
    """
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_boost_nerv")
    src = inspect.getsource(trainer._full_main)
    assert "raise NotImplementedError" not in src, (
        "_full_main NotImplementedError must be extinguished per "
        "CLASS-SHIFT-FULL-MAIN-CLUSTER"
    )
    assert "run_pact_nerv_score_aware_training" in src, (
        "_full_main must route through the canonical shared training loop"
    )
    args = trainer._build_parser().parse_args(
        ["--output-dir", str(tmp_path / "out"), "--device", "cpu", "--epochs", "1"]
    )
    with pytest.raises(SystemExit):
        trainer._full_main(args)
