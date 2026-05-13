"""Tests for tac.scpp_substrate — SC++ self-compression substrate."""
from __future__ import annotations

import struct

import pytest
import torch

from tac.scpp_substrate import (
    ARCHIVE_GRAMMAR,
    PARSER_SECTION_MANIFEST,
    RUNTIME_DEP_CLOSURE,
    SCPP_DEFAULT_BLOCK_SIZE,
    SCPP_DEFAULT_QINT_MAX,
    SCPP_DEFAULT_SIGMA,
    SCPP_FORMAT_ID,
    SCPP_MAGIC,
    SCPP_TARGET_PARAMS_MAX,
    SCPP_TARGET_PARAMS_MIN,
    SCPP_VERSION,
    SCPPSubstrate,
    SCPPSubstrateConfig,
    decode_scpp_substrate,
    encode_scpp_substrate,
    no_op_detect_scpp_archive,
    scpp_archive_bytes_inventory,
)


# ── Config validation ─────────────────────────────────────────────────────


def test_scpp_config_default_construction():
    cfg = SCPPSubstrateConfig()
    assert cfg.latent_dim == 32
    assert cfg.base_channels == 32
    assert cfg.n_pairs == 600
    assert cfg.eval_height == 384
    assert cfg.eval_width == 512
    assert cfg.sigma == SCPP_DEFAULT_SIGMA
    assert cfg.qint_max == SCPP_DEFAULT_QINT_MAX
    assert cfg.block_size == SCPP_DEFAULT_BLOCK_SIZE


def test_scpp_config_rejects_negative_latent_dim():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        SCPPSubstrateConfig(latent_dim=-1)


def test_scpp_config_rejects_zero_base_channels():
    with pytest.raises(ValueError, match="base_channels must be positive"):
        SCPPSubstrateConfig(base_channels=0)


def test_scpp_config_rejects_invalid_qint_max():
    with pytest.raises(ValueError, match="qint_max must be in"):
        SCPPSubstrateConfig(qint_max=8)
    with pytest.raises(ValueError, match="qint_max must be in"):
        SCPPSubstrateConfig(qint_max=0)


def test_scpp_config_rejects_zero_sigma():
    with pytest.raises(ValueError, match="sigma must be positive"):
        SCPPSubstrateConfig(sigma=0)


def test_scpp_config_serialise_round_trip():
    cfg = SCPPSubstrateConfig(latent_dim=8, base_channels=16)
    payload = cfg.serialise()
    restored = SCPPSubstrateConfig.deserialise(payload)
    assert restored == cfg


def test_scpp_config_deserialise_refuses_wrong_version():
    cfg = SCPPSubstrateConfig()
    payload = cfg.serialise()
    payload["format_version"] = 99
    with pytest.raises(ValueError, match="version mismatch"):
        SCPPSubstrateConfig.deserialise(payload)


# ── Substrate architecture ────────────────────────────────────────────────


def test_scpp_substrate_constructs_in_parameter_band():
    """88-94K parameter target per Selfcomp verified config."""
    cfg = SCPPSubstrateConfig()
    sub = SCPPSubstrate(cfg)
    n = sub.count_params()
    assert SCPP_TARGET_PARAMS_MIN <= n <= SCPP_TARGET_PARAMS_MAX, (
        f"Parameter count {n} outside Selfcomp-verified band "
        f"[{SCPP_TARGET_PARAMS_MIN}, {SCPP_TARGET_PARAMS_MAX}]"
    )


def test_scpp_substrate_rejects_out_of_band_config():
    # base_channels=2 + latent_dim=4 + eval=384x512 → way under 80K params
    with pytest.raises(ValueError, match="outside Selfcomp-verified band"):
        SCPPSubstrate(SCPPSubstrateConfig(latent_dim=4, base_channels=2))


def test_scpp_substrate_forward_shape():
    cfg = SCPPSubstrateConfig()
    sub = SCPPSubstrate(cfg)
    latents = torch.randn(2, cfg.latent_dim)
    out = sub(latents)
    assert out.shape == (2, 2, 3, cfg.eval_height, cfg.eval_width)


def test_scpp_substrate_forward_is_differentiable():
    cfg = SCPPSubstrateConfig()
    sub = SCPPSubstrate(cfg)
    latents = torch.randn(2, cfg.latent_dim, requires_grad=True)
    out = sub(latents)
    loss = out.mean()
    loss.backward()
    assert latents.grad is not None
    assert latents.grad.shape == latents.shape


# ── Encoder/decoder round-trip ────────────────────────────────────────────


def test_encode_decode_round_trip_state_dict_approximate():
    """Block-FP encoding is lossy; check that decode-encode round-trip
    is stable (idempotent after the first quantization)."""
    cfg = SCPPSubstrateConfig(latent_dim=8, base_channels=16, n_pairs=4)
    sub = SCPPSubstrate(cfg, unsafe_test_only_skip_param_check=True)
    latents = torch.randn(cfg.n_pairs, cfg.latent_dim) * 0.5

    encoded = encode_scpp_substrate(
        state_dict=sub.state_dict(),
        latents=latents,
        config=cfg,
    )
    sd, lat, cfg_decoded = decode_scpp_substrate(encoded)

    assert cfg_decoded == cfg
    # State dict keys preserved
    assert set(sd.keys()) == set(sub.state_dict().keys())
    # Latents preserved within int8 quantization noise
    assert torch.allclose(lat, latents, atol=0.05)


def test_encode_decode_idempotent_after_first_pass():
    """The decode-encode-decode round trip is bit-stable (identity mode)."""
    cfg = SCPPSubstrateConfig(latent_dim=8, base_channels=16, n_pairs=4)
    sub = SCPPSubstrate(cfg, unsafe_test_only_skip_param_check=True)
    latents = torch.randn(cfg.n_pairs, cfg.latent_dim) * 0.5

    bytes_1 = encode_scpp_substrate(state_dict=sub.state_dict(), latents=latents, config=cfg)
    sd_1, lat_1, cfg_1 = decode_scpp_substrate(bytes_1)
    bytes_2 = encode_scpp_substrate(state_dict=sd_1, latents=lat_1, config=cfg_1)
    assert bytes_1 == bytes_2  # idempotent re-emission


def test_encoded_archive_starts_with_correct_magic_and_format_id():
    cfg = SCPPSubstrateConfig(latent_dim=8, base_channels=16, n_pairs=4)
    sub = SCPPSubstrate(cfg, unsafe_test_only_skip_param_check=True)
    latents = torch.randn(cfg.n_pairs, cfg.latent_dim)
    encoded = encode_scpp_substrate(state_dict=sub.state_dict(), latents=latents, config=cfg)
    magic, format_id, version, _, _, _ = struct.unpack_from("<BBHIII", encoded, 0)
    assert magic == SCPP_MAGIC
    assert format_id == SCPP_FORMAT_ID
    assert version == SCPP_VERSION


def test_decode_rejects_short_archive():
    with pytest.raises(ValueError, match="truncated"):
        decode_scpp_substrate(b"\x00" * 8)


def test_decode_rejects_wrong_magic():
    cfg = SCPPSubstrateConfig(latent_dim=8, base_channels=16, n_pairs=4)
    sub = SCPPSubstrate(cfg, unsafe_test_only_skip_param_check=True)
    encoded = encode_scpp_substrate(
        state_dict=sub.state_dict(),
        latents=torch.randn(cfg.n_pairs, cfg.latent_dim),
        config=cfg,
    )
    # Corrupt magic byte
    corrupted = bytes([0x99]) + encoded[1:]
    with pytest.raises(ValueError, match="magic mismatch"):
        decode_scpp_substrate(corrupted)


def test_decode_rejects_wrong_format_id():
    cfg = SCPPSubstrateConfig(latent_dim=8, base_channels=16, n_pairs=4)
    sub = SCPPSubstrate(cfg, unsafe_test_only_skip_param_check=True)
    encoded = encode_scpp_substrate(
        state_dict=sub.state_dict(),
        latents=torch.randn(cfg.n_pairs, cfg.latent_dim),
        config=cfg,
    )
    corrupted = encoded[:1] + bytes([0x99]) + encoded[2:]
    with pytest.raises(ValueError, match="format_id mismatch"):
        decode_scpp_substrate(corrupted)


def test_decode_rejects_size_mismatch():
    cfg = SCPPSubstrateConfig(latent_dim=8, base_channels=16, n_pairs=4)
    sub = SCPPSubstrate(cfg, unsafe_test_only_skip_param_check=True)
    encoded = encode_scpp_substrate(
        state_dict=sub.state_dict(),
        latents=torch.randn(cfg.n_pairs, cfg.latent_dim),
        config=cfg,
    )
    # Append trailing bytes — decode must reject
    corrupted = encoded + b"\x00\x00\x00\x00"
    with pytest.raises(ValueError, match="size mismatch"):
        decode_scpp_substrate(corrupted)


# ── No-op detector ────────────────────────────────────────────────────────


def test_no_op_detect_identical_bytes():
    cfg = SCPPSubstrateConfig(latent_dim=8, base_channels=16, n_pairs=4)
    sub = SCPPSubstrate(cfg, unsafe_test_only_skip_param_check=True)
    encoded = encode_scpp_substrate(
        state_dict=sub.state_dict(),
        latents=torch.randn(cfg.n_pairs, cfg.latent_dim),
        config=cfg,
    )
    verdict = no_op_detect_scpp_archive(encoded, encoded)
    assert verdict["bytes_changed"] is False
    assert verdict["verdict"] == "no_op"


def test_no_op_detect_changed_latents():
    cfg = SCPPSubstrateConfig(latent_dim=8, base_channels=16, n_pairs=4)
    sub = SCPPSubstrate(cfg, unsafe_test_only_skip_param_check=True)
    sd = sub.state_dict()

    lat1 = torch.randn(cfg.n_pairs, cfg.latent_dim) * 0.5
    lat2 = lat1 + 0.3  # large enough to survive int8 quantization

    enc1 = encode_scpp_substrate(state_dict=sd, latents=lat1, config=cfg)
    enc2 = encode_scpp_substrate(state_dict=sd, latents=lat2, config=cfg)

    assert enc1 != enc2
    verdict = no_op_detect_scpp_archive(enc1, enc2)
    assert verdict["bytes_changed"] is True
    assert verdict["latents_changed"] is True
    assert verdict["verdict"] == "consumed"


# ── Section inventory ─────────────────────────────────────────────────────


def test_archive_bytes_inventory_sums_to_total():
    cfg = SCPPSubstrateConfig(latent_dim=8, base_channels=16, n_pairs=4)
    sub = SCPPSubstrate(cfg, unsafe_test_only_skip_param_check=True)
    encoded = encode_scpp_substrate(
        state_dict=sub.state_dict(),
        latents=torch.randn(cfg.n_pairs, cfg.latent_dim),
        config=cfg,
    )
    inv = scpp_archive_bytes_inventory(encoded)
    assert inv["total_bytes"] == len(encoded)
    assert (
        inv["header_bytes"]
        + inv["config_json_bytes"]
        + inv["blockfp_weights_bytes"]
        + inv["latents_bytes"]
        == inv["total_bytes"]
    )
    assert inv["trailing_bytes"] == 0
    assert inv["magic"] == SCPP_MAGIC
    assert inv["format_id"] == SCPP_FORMAT_ID


def test_inventory_handles_truncated_archive():
    inv = scpp_archive_bytes_inventory(b"\x00" * 8)
    assert inv["error"] == "archive truncated"


# ── Archive grammar constants ─────────────────────────────────────────────


def test_archive_grammar_declares_required_hnerv_parity_fields():
    """Per CLAUDE.md HNeRV parity discipline: representation lanes must
    declare 8 fields at design time (check_gate1 + #124)."""
    required = {
        "archive_grammar",  # implicit via wire_format key
        "no_op_detector_planned",
        "score_aware_loss",
        "inflate_runtime_loc_budget",
        "runtime_dep_closure",
        "export_format",
        "bolt_on_loc_budget",
        "lane_class",
    }
    # ARCHIVE_GRAMMAR contains the necessary fields
    declared = set(ARCHIVE_GRAMMAR.keys())
    missing = required - declared - {"archive_grammar"}  # archive_grammar is the dict itself
    assert not missing, f"Missing HNeRV parity fields: {missing}"
    assert ARCHIVE_GRAMMAR["no_op_detector_planned"] is True
    assert ARCHIVE_GRAMMAR["inflate_runtime_loc_budget"] == 200
    assert ARCHIVE_GRAMMAR["lane_class"] == "substrate_engineering"


def test_runtime_dep_closure_is_minimal():
    """Inflate runtime should depend only on torch + brotli."""
    assert set(RUNTIME_DEP_CLOSURE) == {"torch", "brotli"}


def test_parser_section_manifest_has_all_sections():
    section_names = {s["name"] for s in PARSER_SECTION_MANIFEST}
    assert section_names == {"header", "config_json", "blockfp_weights", "latents"}
    # Header is non-mutable
    header = next(s for s in PARSER_SECTION_MANIFEST if s["name"] == "header")
    assert header["mutable"] is False
    assert header["fixed_offset"] == 0
    assert header["fixed_length"] == 16
