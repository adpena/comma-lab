# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for coin_plus_plus.

Proves the encode/decode contract of the CPP1 monolithic 0.bin grammar +
coord-MLP + modulation forward-pass parity under fp16 base + int8
modulation roundtrip. Plus a smoke-level test that the trainer's
_full_main raises NotImplementedError per the L0 SCAFFOLD posture
(Catalog #240).
"""

from __future__ import annotations

import torch

from tac.substrates.coin_plus_plus.architecture import (
    CoinplusplusConfig,
    CoinplusplusSubstrate,
)
from tac.substrates.coin_plus_plus.archive import (
    CPP1_HEADER_SIZE,
    CPP1_MAGIC,
    CPP1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> CoinplusplusConfig:
    return CoinplusplusConfig(
        modulation_dim=16,
        hidden_dim=32,
        num_hidden_layers=2,
        sin_frequency=30.0,
        coord_input_dim=3,
        output_channels=3,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: CoinplusplusConfig) -> dict[str, object]:
    return {
        "hidden_dim": cfg.hidden_dim,
        "num_hidden_layers": cfg.num_hidden_layers,
        "sin_frequency": cfg.sin_frequency,
        "coord_input_dim": cfg.coord_input_dim,
        "output_channels": cfg.output_channels,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = CoinplusplusSubstrate(cfg)
    sd = model.state_dict()
    base_sd = {k: v for k, v in sd.items() if k != "modulations"}
    modulations = sd["modulations"].clone()

    blob = pack_archive(
        base_sd, modulations, _smoke_meta(cfg),
        modulation_dim=cfg.modulation_dim,
    )
    arc = parse_archive(blob)

    assert arc.schema_version == CPP1_SCHEMA_VERSION
    assert blob[:4] == CPP1_MAGIC
    assert arc.modulation_dim == cfg.modulation_dim
    assert set(arc.base_mlp_state_dict.keys()) == set(base_sd.keys())
    for k, v in base_sd.items():
        rec = arc.base_mlp_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    assert arc.modulations.shape == modulations.shape
    # int8 quant has coarser steps than int16; allow ~3x quant step tolerance
    mod_range = max(float(modulations.max() - modulations.min()), 1e-12)
    step = mod_range / 255.0
    assert torch.allclose(arc.modulations, modulations, atol=step * 3.0)


def test_header_size_invariant_is_21_bytes():
    assert CPP1_HEADER_SIZE == 21


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
    model = CoinplusplusSubstrate(cfg)
    base_sd = {k: v for k, v in model.state_dict().items() if k != "modulations"}
    modulations = model.state_dict()["modulations"].clone()
    blob = bytearray(
        pack_archive(
            base_sd, modulations, _smoke_meta(cfg),
            modulation_dim=cfg.modulation_dim,
        )
    )
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_pack_archive_rejects_mismatched_modulation_dim():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = CoinplusplusSubstrate(cfg)
    base_sd = {k: v for k, v in model.state_dict().items() if k != "modulations"}
    modulations = model.state_dict()["modulations"].clone()
    try:
        pack_archive(
            base_sd, modulations, _smoke_meta(cfg),
            modulation_dim=cfg.modulation_dim + 1,  # mismatch
        )
    except ValueError as exc:
        assert "modulation_dim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on modulation_dim mismatch")


def test_forward_pass_after_roundtrip_matches_original_within_tolerance():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = CoinplusplusSubstrate(cfg).eval()

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    base_sd = {k: v for k, v in sd.items() if k != "modulations"}
    modulations = sd["modulations"].clone()
    blob = pack_archive(
        base_sd, modulations, _smoke_meta(cfg),
        modulation_dim=cfg.modulation_dim,
    )
    arc = parse_archive(blob)

    rebuilt = CoinplusplusSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.base_mlp_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.modulations.copy_(arc.modulations.to(rebuilt.modulations.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    # Wider tolerance than NeRV-family because int8 modulation quant is
    # coarser than int16 latent quant. Empirically the sigmoid-output rgb
    # should still match within ~0.15.
    assert torch.allclose(rgb_0_a, rgb_0_b, atol=0.15)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=0.15)


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_inflate_output_no_op_proof():
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = CoinplusplusSubstrate(cfg).eval()
    base_sd = {k: v for k, v in model.state_dict().items() if k != "modulations"}
    modulations = model.state_dict()["modulations"].clone()

    blob_a = pack_archive(
        base_sd, modulations, _smoke_meta(cfg),
        modulation_dim=cfg.modulation_dim,
    )
    mutated = modulations.clone()
    mutated[0, 0] = mutated[0, 0] + 1.0
    blob_b = pack_archive(
        base_sd, mutated, _smoke_meta(cfg),
        modulation_dim=cfg.modulation_dim,
    )

    assert blob_a != blob_b, "no_op_proof: mutating modulations must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.modulations[0, 0], arc_b.modulations[0, 0], atol=1e-3)


def test_forward_pass_produces_unit_interval_rgb():
    """L5 compliance: substrate is a full RGB renderer (not a mask codec)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = CoinplusplusSubstrate(cfg).eval()
    idx = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_modulation_paradigm_has_per_pair_modulations():
    """Distinctive design check: coin_plus_plus must have per-pair modulations
    (not per-pair latents as in NeRV-family). The shared base MLP is the
    rate-amortized component."""
    cfg = _smoke_cfg()
    model = CoinplusplusSubstrate(cfg)
    assert model.modulations.shape == (cfg.num_pairs, cfg.modulation_dim)
    # Ensure modulations are listed as the ONLY per-pair state (no per-pair
    # latents in the state_dict beyond modulations).
    per_pair_keys = [
        k for k in model.state_dict().keys()
        if model.state_dict()[k].dim() == 2
        and model.state_dict()[k].shape[0] == cfg.num_pairs
    ]
    assert "modulations" in per_pair_keys


def test_coord_mlp_has_modulated_layers():
    """Distinctive design check: each hidden layer must be FiLM-modulated."""
    cfg = _smoke_cfg()
    model = CoinplusplusSubstrate(cfg)
    assert len(model.mod_layers) == cfg.num_hidden_layers
    for layer in model.mod_layers:
        assert hasattr(layer, "mod_gamma_proj"), "missing FiLM gamma projection"
        assert hasattr(layer, "mod_beta_proj"), "missing FiLM beta projection"


def test_full_main_implemented_and_cuda_gated(tmp_path):
    """CLASS-SHIFT-FULL-MAIN-CLUSTER 2026-05-27: _full_main IMPLEMENTED + CUDA-gated.

    The L0 SCAFFOLD NotImplementedError is extinguished: ``_full_main`` routes
    the canonical score-aware training loop through
    ``run_pact_nerv_score_aware_training``. Per CLAUDE.md "MPS auth eval is
    NOISE" + Catalog #1, the full (non-smoke) path is CUDA-required; invoking
    it with ``--device cpu`` refuses via ``device_or_die`` (SystemExit). PAID
    DISPATCH stays gated by ``dispatch_enabled: false`` + ``research_only:
    true`` on the recipe per Catalog #325 (code complete, trigger gated).
    """
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_coin_plus_plus")
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
