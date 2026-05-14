# SPDX-License-Identifier: MIT
"""Dedicated tests for the S2SBS substrate (Council F O3, 2026-05-13)."""

from __future__ import annotations

import pytest
import torch

from tac.substrates.s2sbs_byte_stuffing import (
    S2S1_GRAMMAR,
    S2S1_HEADER_STRUCT,
    S2S1_SCHEMA_VERSION,
    S2SBS_AR_MAGIC,
    S2SBS_METADATA,
    HfBlindspotMask,
    HfFftByteCodec,
    PayloadChannel,
    S2sbsArchive,
    S2sbsConfig,
    S2sbsLossWeights,
    S2sbsRenderer,
    S2sbsScoreAwareLoss,
    pack_archive,
    parse_archive,
)
from tac.substrates.s2sbs_byte_stuffing.architecture import (
    DEFAULT_DELTA_AMP_UINT8,
    MAX_DELTA_AMP_UINT8_BEFORE_POSE_DRIFT,
)
from tac.substrates.s2sbs_byte_stuffing.inflate import (
    inflate_one_video,
    render_pair,
)


def _cfg(**overrides) -> S2sbsConfig:
    base = {
        "num_pairs": 2,
        "output_height": 16,
        "output_width": 24,
        "hf_blindspot_lf_cutoff_h": 4,
        "hf_blindspot_lf_cutoff_w": 6,
        "delta_amp_uint8": 0.75,
        "payload_channel": "R",
        "base_seed": 7,
        "payload_bytes_per_pair": 8,
        "ecc_rate": 0.25,
    }
    base.update(overrides)
    return S2sbsConfig(**base)


# --------------------- Config validation ---------------------------


def test_config_defaults_match_audit_pinned_values() -> None:
    cfg = S2sbsConfig()
    assert cfg.num_pairs == 600
    assert cfg.output_height == 384
    assert cfg.output_width == 512
    assert cfg.delta_amp_uint8 == DEFAULT_DELTA_AMP_UINT8
    assert cfg.payload_channel == "R"
    assert cfg.channel_index == 0


def test_config_rejects_delta_amp_above_joint_safety_cap() -> None:
    with pytest.raises(ValueError, match="delta_amp_uint8"):
        S2sbsConfig(delta_amp_uint8=MAX_DELTA_AMP_UINT8_BEFORE_POSE_DRIFT + 0.01)


def test_config_rejects_zero_delta_amp() -> None:
    with pytest.raises(ValueError, match="delta_amp_uint8"):
        S2sbsConfig(delta_amp_uint8=0.0)


def test_config_rejects_invalid_channel() -> None:
    with pytest.raises(ValueError, match="payload_channel"):
        S2sbsConfig(payload_channel="X")


def test_config_rejects_invalid_lf_cutoff() -> None:
    with pytest.raises(ValueError, match="hf_blindspot_lf_cutoff_h"):
        S2sbsConfig(output_height=8, hf_blindspot_lf_cutoff_h=8)


def test_config_rejects_invalid_ecc_rate() -> None:
    with pytest.raises(ValueError, match="ecc_rate"):
        S2sbsConfig(ecc_rate=1.5)


def test_raw_payload_bytes_per_pair_inflates_by_ecc_overhead() -> None:
    cfg = _cfg(payload_bytes_per_pair=16, ecc_rate=0.25)
    assert cfg.raw_payload_bytes_per_pair == 64


# ----------------------- HF blindspot mask -------------------------


def test_blindspot_mask_excludes_dc_and_nyquist_lines() -> None:
    cfg = _cfg()
    mask = HfBlindspotMask(cfg).mask
    assert mask[0, 0].item() is False
    assert mask[cfg.output_height // 2, :].any().item() is False
    assert mask[:, cfg.output_width // 2].any().item() is False


def test_blindspot_mask_has_nonzero_capacity() -> None:
    cfg = _cfg()
    mask_mod = HfBlindspotMask(cfg)
    assert mask_mod.coordinate_count() > 0


# ---------------------- Hermitian-FFT codec ------------------------


def test_codec_roundtrip_exact_at_zero_noise() -> None:
    cfg = _cfg()
    codec = HfFftByteCodec(cfg)
    base_channel = torch.full(
        (cfg.output_height, cfg.output_width), 0.5, dtype=torch.float32
    )
    payload = b"hello"
    encoded = codec.encode(base_channel, payload)
    recovered = codec.decode(encoded, base_channel)
    assert recovered[: len(payload)] == payload


def test_codec_capacity_matches_blindspot_count() -> None:
    cfg = _cfg()
    codec = HfFftByteCodec(cfg)
    assert codec.capacity_bits == HfBlindspotMask(cfg).coordinate_count()
    assert codec.capacity_bytes == codec.capacity_bits // 8


def test_codec_refuses_oversized_payload() -> None:
    cfg = _cfg()
    codec = HfFftByteCodec(cfg)
    base = torch.zeros(cfg.output_height, cfg.output_width)
    too_big = b"\xff" * (codec.capacity_bytes + 1)
    with pytest.raises(ValueError, match="capacity"):
        codec.encode(base, too_big)


def test_codec_modifies_only_configured_channel_when_3d() -> None:
    cfg = _cfg(payload_channel="G")
    codec = HfFftByteCodec(cfg)
    base = torch.full((3, cfg.output_height, cfg.output_width), 0.5)
    encoded = codec.encode(base, b"\x55")
    assert torch.equal(encoded[0], base[0])  # R untouched
    assert torch.equal(encoded[2], base[2])  # B untouched
    assert not torch.equal(encoded[1], base[1])  # G payload


# ----------------------- Renderer ----------------------------------


def test_renderer_base_pair_deterministic() -> None:
    cfg = _cfg()
    renderer = S2sbsRenderer(cfg)
    pair_indices = torch.tensor([0, 1], dtype=torch.long)
    rgb0_a, rgb1_a = renderer(pair_indices)
    rgb0_b, rgb1_b = renderer(pair_indices)
    assert torch.allclose(rgb0_a, rgb0_b)
    assert torch.allclose(rgb1_a, rgb1_b)
    assert rgb0_a.shape == (2, 3, cfg.output_height, cfg.output_width)


def test_renderer_payload_changes_output_only_on_targeted_pair() -> None:
    cfg = _cfg()
    renderer = S2sbsRenderer(cfg)
    pair_indices = torch.tensor([0, 1], dtype=torch.long)
    rgb0_base, _ = renderer(pair_indices)
    rgb0_paid, _ = renderer(
        pair_indices,
        payload_by_pair=(PayloadChannel(pair_index=0, payload=b"\xAA"),),
    )
    assert not torch.allclose(rgb0_base[0], rgb0_paid[0])
    assert torch.allclose(rgb0_base[1], rgb0_paid[1])


def test_renderer_refuses_empty_or_wrong_dtype() -> None:
    cfg = _cfg()
    renderer = S2sbsRenderer(cfg)
    with pytest.raises(ValueError, match=r"pair_indices must be torch\.long"):
        renderer(torch.tensor([0], dtype=torch.int32))
    with pytest.raises(ValueError, match="non-empty"):
        renderer(torch.tensor([], dtype=torch.long))


# ----------------------- Archive grammar ---------------------------


def _payloads() -> tuple[PayloadChannel, ...]:
    return (
        PayloadChannel(pair_index=0, payload=b"\x01\x02\x03\x04"),
        PayloadChannel(pair_index=1, payload=b"\xDE\xAD\xBE\xEF"),
    )


def test_archive_roundtrip_is_deterministic_and_charges_bytes() -> None:
    cfg = _cfg()
    blob_a = pack_archive(config=cfg, payloads=_payloads())
    blob_b = pack_archive(config=cfg, payloads=_payloads())
    assert blob_a == blob_b
    assert blob_a[:4] == S2SBS_AR_MAGIC
    assert blob_a[4:8] == S2S1_GRAMMAR
    assert blob_a[8] == S2S1_SCHEMA_VERSION
    assert len(blob_a) > S2S1_HEADER_STRUCT.size

    parsed = parse_archive(blob_a)
    assert isinstance(parsed, S2sbsArchive)
    assert parsed.config == cfg
    assert parsed.charged_bytes == len(blob_a)
    assert parsed.total_payload_bytes == 8
    assert parsed.score_claim is False


def test_byte_mutation_changes_archive_state_and_render_smoke() -> None:
    cfg = _cfg()
    blob_a = pack_archive(config=cfg, payloads=_payloads())
    blob_b = pack_archive(
        config=cfg,
        payloads=(
            PayloadChannel(pair_index=0, payload=b"\x01\x02\x03\x05"),  # one byte flipped
            PayloadChannel(pair_index=1, payload=b"\xDE\xAD\xBE\xEF"),
        ),
    )
    assert blob_a != blob_b
    parsed_a = parse_archive(blob_a)
    parsed_b = parse_archive(blob_b)
    assert parsed_a.payloads[0].payload != parsed_b.payloads[0].payload
    rgb0_a, _ = render_pair(parsed_a, 0, device="cpu")
    rgb0_b, _ = render_pair(parsed_b, 0, device="cpu")
    assert not torch.allclose(rgb0_a, rgb0_b)


def test_archive_parse_fails_closed_on_corruption() -> None:
    cfg = _cfg()
    blob = bytearray(pack_archive(config=cfg, payloads=_payloads()))
    with pytest.raises(ValueError, match="too short"):
        parse_archive(b"x")
    bad_magic = bytearray(blob)
    bad_magic[:4] = b"NOPE"
    with pytest.raises(ValueError, match="magic"):
        parse_archive(bytes(bad_magic))
    bad_checksum = bytearray(blob)
    bad_checksum[-1] ^= 0x01
    with pytest.raises(ValueError, match="checksum"):
        parse_archive(bytes(bad_checksum))
    with pytest.raises(ValueError, match="header-declared"):
        parse_archive(bytes(blob) + b"extra")


def test_archive_refuses_metadata_with_score_claim_true() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match="score_claim"):
        pack_archive(config=cfg, payloads=(), metadata={"score_claim": True})


def test_archive_refuses_payload_overflowing_num_pairs() -> None:
    cfg = _cfg(num_pairs=2)
    overflow = PayloadChannel(pair_index=5, payload=b"x")
    with pytest.raises(ValueError, match="num_pairs"):
        pack_archive(config=cfg, payloads=(overflow,))


def test_archive_metadata_forces_research_only_false_authority() -> None:
    blob = pack_archive(config=_cfg(), payloads=_payloads())
    parsed = parse_archive(blob)
    assert parsed.metadata["research_only"] is True
    for key in (
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "scorer_authority",
    ):
        assert parsed.metadata[key] is False


def test_module_metadata_carries_audit_evidence_tag() -> None:
    assert S2SBS_METADATA["research_only"] is True
    assert S2SBS_METADATA["score_claim"] is False
    assert S2SBS_METADATA["audit_evidence_grade"] == "macOS-CPU advisory"


# ------------------ Inflate smoke ----------------------------------


def test_inflate_smoke_writes_deterministic_raw(tmp_path) -> None:
    blob = pack_archive(config=_cfg(), payloads=_payloads())
    raw_a = tmp_path / "a.raw"
    raw_b = tmp_path / "b.raw"
    n_a = inflate_one_video(blob, raw_a, device="cpu")
    n_b = inflate_one_video(blob, raw_b, device="cpu")
    assert n_a == _cfg().num_pairs * 2
    assert n_a == n_b
    assert raw_a.read_bytes() == raw_b.read_bytes()


def test_inflate_refuses_non_cpu_device(tmp_path) -> None:
    blob = pack_archive(config=_cfg(), payloads=())
    with pytest.raises(RuntimeError, match="CPU-only"):
        inflate_one_video(blob, tmp_path / "x.raw", device="cuda")


# ---------- Score-aware loss & Catalog discipline ------------------


class _PreprocessScorer(torch.nn.Module):
    """Minimal scorer with preprocess_input + forward returning logits-like tensor."""

    def __init__(self, *, out_channels: int = 5) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.conv = torch.nn.Conv2d(3, out_channels, kernel_size=3, padding=1)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        # Reduce to the last frame at 4D contract.
        return pair_btchw[:, -1]

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.conv(x_bchw)


class _PosePreprocessScorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(12, 6)

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = pair_btchw.shape
        flat = pair_btchw.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
        flat6 = flat.expand(-1, 6, -1, -1)
        half_h = h // 2
        half_w = w // 2
        flat6_sub = flat6[..., : half_h * 2, : half_w * 2].reshape(
            b * t, 6, half_h, 2, half_w, 2
        ).mean(dim=(3, 5))
        return flat6_sub.reshape(b, t * 6, half_h, half_w)

    def forward(self, x_b12hw: torch.Tensor) -> torch.Tensor:
        pooled = x_b12hw.flatten(2).mean(dim=2)
        return self.head(pooled)


def test_score_aware_loss_refuses_eval_roundtrip_false() -> None:
    loss_mod = S2sbsScoreAwareLoss(
        _PreprocessScorer(out_channels=5),
        _PosePreprocessScorer(),
        S2sbsLossWeights(),
    )
    rgb = torch.zeros(1, 3, 16, 24)
    with pytest.raises(ValueError, match="eval_roundtrip"):
        loss_mod(
            rgb, rgb, rgb, rgb, torch.tensor(128.0), apply_eval_roundtrip=False
        )


def test_score_aware_loss_weights_validate() -> None:
    with pytest.raises(ValueError, match="alpha_rate"):
        S2sbsLossWeights(alpha_rate=-0.1)
    with pytest.raises(ValueError, match="contest_normalizer"):
        S2sbsLossWeights(contest_normalizer=0.0)


# ---------- Capacity sanity -----------------------------------------


def test_codec_capacity_grows_with_hf_band() -> None:
    cfg_small_band = _cfg(hf_blindspot_lf_cutoff_h=6, hf_blindspot_lf_cutoff_w=8)
    cfg_large_band = _cfg(hf_blindspot_lf_cutoff_h=3, hf_blindspot_lf_cutoff_w=4)
    capacity_small = HfFftByteCodec(cfg_small_band).capacity_bits
    capacity_large = HfFftByteCodec(cfg_large_band).capacity_bits
    assert capacity_large > capacity_small


def test_payload_channel_rejects_invalid_pair_index() -> None:
    with pytest.raises(ValueError, match="pair_index"):
        PayloadChannel(pair_index=-1)


def test_payload_channel_rejects_oversized_payload() -> None:
    with pytest.raises(ValueError, match="uint16"):
        PayloadChannel(pair_index=0, payload=b"\x00" * 70000)
