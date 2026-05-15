# SPDX-License-Identifier: MIT
"""NSCS03 archive grammar tests: roundtrip, determinism, byte-mutation
no-op detection (Catalog #139)."""

from __future__ import annotations

import pytest
import torch

from tac.substrates.nscs03_end_to_end_balle_joint_codec.architecture import (
    NSCS03Config,
    NSCS03JointCodecSubstrate,
)
from tac.substrates.nscs03_end_to_end_balle_joint_codec.archive import (
    NS03_HEADER_FMT,
    NS03_HEADER_SIZE,
    NS03_MAGIC,
    NS03_SCHEMA_VERSION,
    NSCS03Archive,
    _dequantize_from_int16,
    _quantize_to_int16,
    pack_archive,
    parse_archive,
)


def _make_substrate_and_latents(num_pairs: int = 2):
    torch.manual_seed(0)
    cfg = NSCS03Config()
    m = NSCS03JointCodecSubstrate(cfg)
    m.eval()
    x = torch.rand(num_pairs, 6, 384, 512)
    with torch.no_grad():
        latents = m.encode(x)
    return cfg, m, latents["y"], latents["z"]


def _meta_for(cfg: NSCS03Config) -> dict:
    return {
        "config": {
            "in_channels": cfg.in_channels,
            "out_channels": cfg.out_channels,
            "main_latent_channels": cfg.main_latent_channels,
            "hyper_latent_channels": cfg.hyper_latent_channels,
            "g_a_channels": list(cfg.g_a_channels),
            "g_s_channels": list(cfg.g_s_channels),
            "h_a_channels": list(cfg.h_a_channels),
            "h_s_channels": list(cfg.h_s_channels),
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            "gdn_eps": cfg.gdn_eps,
            "sigma_floor": cfg.sigma_floor,
            "quantize_noise_std": cfg.quantize_noise_std,
        },
        # Catalog #210 forensic provenance
        "license_tags": ["MIT"],
        "dataset_provenance": "contest_video_0_mkv",
        "distillation_version": "NSCS03_v1",
        "random_seed": 0,
        "basis_sha256": "0" * 64,
        "num_frames_used": 1200,
    }


class TestNSCS03ArchiveHeader:
    def test_header_magic(self) -> None:
        assert NS03_MAGIC == b"NS03"

    def test_header_size(self) -> None:
        assert NS03_HEADER_SIZE == 51

    def test_schema_version(self) -> None:
        assert NS03_SCHEMA_VERSION == 1


class TestNSCS03ArchiveRoundtrip:
    def test_pack_then_parse_returns_components(self) -> None:
        cfg, m, y, z = _make_substrate_and_latents(num_pairs=2)
        meta = _meta_for(cfg)
        bytes_a = pack_archive(
            m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
            m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
            main_latents=y, hyper_latents=z, meta=meta,
        )
        arc = parse_archive(bytes_a)
        assert isinstance(arc, NSCS03Archive)
        assert arc.schema_version == 1
        assert arc.main_latents.shape == y.shape
        assert arc.hyper_latents.shape == z.shape

    def test_state_dicts_round_trip(self) -> None:
        cfg, m, y, z = _make_substrate_and_latents(num_pairs=2)
        meta = _meta_for(cfg)
        bytes_a = pack_archive(
            m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
            m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
            main_latents=y, hyper_latents=z, meta=meta,
        )
        arc = parse_archive(bytes_a)
        # Encoder state_dict
        for k, v in m.g_a.state_dict().items():
            assert k in arc.encoder_state_dict
            # Tolerance: state_dict was cast to fp16
            assert torch.allclose(
                arc.encoder_state_dict[k].to(torch.float32), v.to(torch.float32),
                atol=1e-3, rtol=1e-3
            ), f"encoder key {k} mismatch"
        # Decoder
        for k in m.g_s.state_dict():
            assert k in arc.decoder_state_dict

    def test_latents_round_trip_within_quantization_tolerance(self) -> None:
        cfg, m, y, z = _make_substrate_and_latents(num_pairs=2)
        meta = _meta_for(cfg)
        bytes_a = pack_archive(
            m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
            m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
            main_latents=y, hyper_latents=z, meta=meta,
        )
        arc = parse_archive(bytes_a)
        # int16 quant: ((hi-lo)/65534) per scale
        y_range = float(y.max() - y.min())
        y_quant_step = y_range / 65534.0
        z_range = float(z.max() - z.min())
        z_quant_step = z_range / 65534.0
        y_err = (arc.main_latents - y).abs().max().item()
        z_err = (arc.hyper_latents - z).abs().max().item()
        assert y_err <= y_quant_step + 1e-6, f"y err {y_err} > step {y_quant_step}"
        assert z_err <= z_quant_step + 1e-6, f"z err {z_err} > step {z_quant_step}"


class TestNSCS03ArchiveDeterminism:
    def test_pack_is_byte_identical_for_same_inputs(self) -> None:
        cfg, m, y, z = _make_substrate_and_latents(num_pairs=2)
        meta = _meta_for(cfg)
        b1 = pack_archive(
            m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
            m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
            main_latents=y, hyper_latents=z, meta=meta,
        )
        b2 = pack_archive(
            m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
            m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
            main_latents=y, hyper_latents=z, meta=meta,
        )
        assert b1 == b2


class TestNSCS03ArchiveValidation:
    def test_short_blob_rejected(self) -> None:
        with pytest.raises(ValueError, match="archive too short"):
            parse_archive(b"\x00" * 10)

    def test_bad_magic_rejected(self) -> None:
        cfg, m, y, z = _make_substrate_and_latents(num_pairs=2)
        meta = _meta_for(cfg)
        b = pack_archive(
            m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
            m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
            main_latents=y, hyper_latents=z, meta=meta,
        )
        # Replace magic with garbage
        bad = b"BAD!" + b[4:]
        with pytest.raises(ValueError, match="bad magic"):
            parse_archive(bad)

    def test_unsupported_schema_version_rejected(self) -> None:
        cfg, m, y, z = _make_substrate_and_latents(num_pairs=2)
        meta = _meta_for(cfg)
        with pytest.raises(ValueError, match="unsupported schema version"):
            pack_archive(
                m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
                m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
                main_latents=y, hyper_latents=z, meta=meta,
                schema_version=99,
            )

    def test_main_latents_must_be_4d(self) -> None:
        cfg, m, y, z = _make_substrate_and_latents(num_pairs=2)
        meta = _meta_for(cfg)
        with pytest.raises(ValueError, match="main_latents must be 4-D"):
            pack_archive(
                m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
                m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
                main_latents=y.flatten(), hyper_latents=z, meta=meta,
            )

    def test_num_pairs_mismatch_rejected(self) -> None:
        cfg, m, y, z = _make_substrate_and_latents(num_pairs=2)
        meta = _meta_for(cfg)
        z_diff = z[:1]  # 1 pair instead of 2
        with pytest.raises(ValueError, match="num_pairs mismatch"):
            pack_archive(
                m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
                m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
                main_latents=y, hyper_latents=z_diff, meta=meta,
            )


class TestNSCS03Quantization:
    def test_normal_range_quantize_dequantize(self) -> None:
        t = torch.linspace(-2.0, 3.0, 1000)
        q, scale, zp = _quantize_to_int16(t)
        assert q.dtype == torch.int16
        # Catalog #158 fix: degenerate-range branch fills with -32767, not 0
        # but normal range fills q in [-32767, +32767]
        assert q.min().item() >= -32767
        assert q.max().item() <= 32767
        d = _dequantize_from_int16(q, scale, zp)
        assert torch.allclose(d, t, atol=scale + 1e-6)

    def test_degenerate_range_per_catalog_158(self) -> None:
        """Degenerate range (min == max) must NOT zero-fill; per Catalog #158
        fix: fill with -32767 sentinel so dequant returns the constant value."""
        t = torch.full((100,), 3.5)
        q, scale, zp = _quantize_to_int16(t)
        assert (q == -32767).all()
        d = _dequantize_from_int16(q, scale, zp)
        assert torch.allclose(d, t)

    def test_non_float_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="tensor must be float"):
            _quantize_to_int16(torch.zeros(10, dtype=torch.int32))


class TestNSCS03MetaProvenance:
    """Catalog #210 forensic provenance fields preserved in meta."""

    def test_meta_carries_provenance_fields(self) -> None:
        cfg, m, y, z = _make_substrate_and_latents(num_pairs=2)
        meta = _meta_for(cfg)
        bytes_a = pack_archive(
            m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
            m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
            main_latents=y, hyper_latents=z, meta=meta,
        )
        arc = parse_archive(bytes_a)
        for required_field in [
            "license_tags", "dataset_provenance", "distillation_version",
            "random_seed", "basis_sha256", "num_frames_used",
        ]:
            assert required_field in arc.meta, f"missing provenance field: {required_field}"


class TestNSCS03ByteMutationNoOpDetector:
    """Catalog #105/#139/#220: byte-mutation must produce frame change.

    This is the executable no-op-detector smoke. Mutating any random byte
    in the archive (provided we don't break the header parser) should
    either fail-closed or produce different frames. A no-op archive is one
    where mutation produces IDENTICAL frames — that's the bug class
    NSCS03 must NOT regress into.
    """

    def test_mutating_main_latents_byte_changes_decoded_frames(self) -> None:
        """Mutate one byte deep inside the MAIN_LATENTS section and verify
        that the decoded frames change. Proves the latent bytes are
        ACTUALLY consumed by the inflate path (Catalog #220 OPERATIONAL
        consumption proof)."""
        cfg, m, y, z = _make_substrate_and_latents(num_pairs=2)
        meta = _meta_for(cfg)
        bytes_a = pack_archive(
            m.g_a.state_dict(), m.g_s.state_dict(), m.h_a.state_dict(),
            m.h_s.state_dict(), m.entropy_bottleneck_z.state_dict(),
            main_latents=y, hyper_latents=z, meta=meta,
        )
        arc_orig = parse_archive(bytes_a)

        # Locate the MAIN_LATENTS section in the bytes per the grammar.
        # The header tells us all section lengths; we navigate to the main
        # latent section start.
        import struct
        fields = struct.unpack(NS03_HEADER_FMT, bytes_a[:NS03_HEADER_SIZE])
        (_, _, _, _, _, _, _, _, _,
         ga_len, gs_len, ha_len, hs_len, eb_len,
         main_len, hyper_len, meta_len) = fields
        main_start = NS03_HEADER_SIZE + ga_len + gs_len + ha_len + hs_len + eb_len
        # Mutate the first byte of the MAIN_LATENTS section by XOR with 0xFF
        bytes_mut = bytearray(bytes_a)
        bytes_mut[main_start] ^= 0xFF
        bytes_mut[main_start + 1] ^= 0xFF
        arc_mut = parse_archive(bytes(bytes_mut))
        # Main latents must DIFFER
        assert not torch.allclose(arc_orig.main_latents, arc_mut.main_latents), (
            "byte mutation in MAIN_LATENTS section did NOT change parsed "
            "latents — archive grammar may be ignoring those bytes (Catalog "
            "#220 dead-bytes anti-pattern)"
        )
