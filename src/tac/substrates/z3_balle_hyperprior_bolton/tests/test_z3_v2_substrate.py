# SPDX-License-Identifier: MIT
"""Tests for Z3 v2 latent-replacement substrate (council omnibus Decision 3).

Tests cover:
- archive_v2.py: encode/decode roundtrip + magic + truncation refusals
- inflate_v2.py: v2-vs-v1-vs-A1 dispatch + latent reconstruction
- end-to-end: build v2 payload + split + decode + reconstruct latents,
  proving that the byte savings are real (Z3HV2 section < A1_LATENT_BLOB_LEN
  is achievable) and that the reconstructed latents are valid fp32 of the
  expected shape.
"""
from __future__ import annotations

import struct

import pytest
import torch

from tac.substrates.z3_balle_hyperprior_bolton.architecture import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    Z3HyperpriorConfig,
    Z3HyperpriorMLP,
)
from tac.substrates.z3_balle_hyperprior_bolton.archive import (
    quantize_int8_with_scale,
)
from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
    A1_DECODER_SECTION_TOTAL,
    A1_LATENT_BLOB_LEN,
    Z3HV2_HEADER_STRUCT,
    Z3HV2_MAGIC,
    Z3HV2_PER_DIM_AFFINE_LEN,
    Z3V2CompositionArchiveContract,
    build_z3v2_composition_archive_contract,
    build_z3v2_payload_bytes,
    decode_z3hv2_section,
    encode_z3hv2_section,
    split_z3v2_payload_bytes,
)
from tac.substrates.z3_balle_hyperprior_bolton.inflate_v2 import (
    is_v2_payload,
    reconstruct_a1_latents,
    reconstruct_a1_latents_from_v2_payload,
    select_inflate_device,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_synthetic_a1_bytes(*, sidecar_size: int = 607) -> bytes:
    """Build minimal A1-shaped bytes: prefix + decoder + latent + sidecar."""
    prefix = struct.pack("<I", A1_DECODER_SECTION_TOTAL)
    decoder = b"D" * 162_164
    latent = b"L" * A1_LATENT_BLOB_LEN
    sidecar = b"S" * sidecar_size
    return prefix + decoder + latent + sidecar


def _build_synthetic_z3hv2_section(
    *,
    n_pairs: int = A1_N_PAIRS,
    hyper_dim: int = 8,
    latent_dim: int = A1_LATENT_DIM,
    seed: int = 0,
) -> tuple[bytes, dict]:
    """Build a deterministic Z3HV2 section + meta dict for round-trip tests."""
    torch.manual_seed(seed)
    cfg = Z3HyperpriorConfig(hyper_latent_dim=hyper_dim)
    mlp = Z3HyperpriorMLP(cfg)
    weight_tensors = torch.cat([p.detach().flatten() for p in mlp.parameters()])
    weights_int8, w_scale = quantize_int8_with_scale(weight_tensors)
    w_hat_int8 = bytes(((i * 13) % 251 - 125) & 0xFF for i in range(n_pairs * hyper_dim))
    residual_int8 = bytes(
        ((i * 7) % 251 - 125) & 0xFF for i in range(n_pairs * latent_dim)
    )
    latent_min = torch.linspace(-1.0, 1.0, latent_dim).to(torch.float32)
    latent_scale = torch.full((latent_dim,), 0.5, dtype=torch.float32)
    section = encode_z3hv2_section(
        hyperprior_weights_int8=weights_int8,
        w_hat_int8=w_hat_int8,
        residual_int8=residual_int8,
        latent_min=latent_min,
        latent_scale=latent_scale,
        hyper_dim=hyper_dim,
        int8_w_scale=w_scale,
        quant_step=cfg.quantization_step,
        min_sigma=cfg.min_sigma,
        max_sigma=cfg.max_sigma,
        factorized_half_range=16.0,
        n_pairs=n_pairs,
        latent_dim=latent_dim,
    )
    return section, {
        "weights_int8": weights_int8,
        "w_hat_int8": w_hat_int8,
        "residual_int8": residual_int8,
        "latent_min": latent_min,
        "latent_scale": latent_scale,
        "int8_w_scale": w_scale,
        "hyper_dim": hyper_dim,
    }


# ---------------------------------------------------------------------------
# archive_v2.py roundtrip tests
# ---------------------------------------------------------------------------


def test_z3hv2_section_starts_with_canonical_magic():
    section, _ = _build_synthetic_z3hv2_section()
    assert section[: len(Z3HV2_MAGIC)] == Z3HV2_MAGIC


def test_z3hv2_section_roundtrip_decodes_all_fields():
    section, meta_in = _build_synthetic_z3hv2_section()
    meta, weights, w_hat, residual, latent_min, latent_scale, _ = decode_z3hv2_section(
        section
    )
    assert meta.n_pairs == A1_N_PAIRS
    assert meta.hyper_dim == meta_in["hyper_dim"]
    assert meta.latent_dim == A1_LATENT_DIM
    assert weights == meta_in["weights_int8"]
    assert w_hat == meta_in["w_hat_int8"]
    assert residual == meta_in["residual_int8"]
    assert torch.allclose(latent_min, meta_in["latent_min"])
    assert torch.allclose(latent_scale, meta_in["latent_scale"])


def test_z3hv2_decode_bad_magic_raises():
    section, _ = _build_synthetic_z3hv2_section()
    bad = b"XXXX" + section[4:]
    with pytest.raises(ValueError, match="bad Z3HV2 magic"):
        decode_z3hv2_section(bad)


def test_z3hv2_decode_truncated_header_raises():
    with pytest.raises(ValueError, match="too short"):
        decode_z3hv2_section(b"Z3V2\x00")


def test_z3hv2_decode_truncated_residual_raises():
    section, _ = _build_synthetic_z3hv2_section()
    truncated = section[:-50]
    with pytest.raises(ValueError, match="truncated"):
        decode_z3hv2_section(truncated)


def test_z3hv2_encode_refuses_wrong_n_pairs():
    section_meta = _build_synthetic_z3hv2_section()[1]
    with pytest.raises(ValueError, match="n_pairs must be"):
        encode_z3hv2_section(
            hyperprior_weights_int8=section_meta["weights_int8"],
            w_hat_int8=section_meta["w_hat_int8"],
            residual_int8=section_meta["residual_int8"],
            latent_min=section_meta["latent_min"],
            latent_scale=section_meta["latent_scale"],
            hyper_dim=8,
            int8_w_scale=1.0,
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
            factorized_half_range=16.0,
            n_pairs=42,
            latent_dim=A1_LATENT_DIM,
        )


def test_z3hv2_encode_refuses_wrong_latent_dim():
    section_meta = _build_synthetic_z3hv2_section()[1]
    bad_latent_min = torch.zeros(7, dtype=torch.float32)
    bad_latent_scale = torch.ones(7, dtype=torch.float32)
    with pytest.raises(ValueError, match="latent_dim must be"):
        encode_z3hv2_section(
            hyperprior_weights_int8=section_meta["weights_int8"],
            w_hat_int8=section_meta["w_hat_int8"],
            residual_int8=section_meta["residual_int8"],
            latent_min=bad_latent_min,
            latent_scale=bad_latent_scale,
            hyper_dim=8,
            int8_w_scale=1.0,
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
            factorized_half_range=16.0,
            latent_dim=7,
        )


def test_z3hv2_encode_refuses_w_hat_length_mismatch():
    section_meta = _build_synthetic_z3hv2_section()[1]
    with pytest.raises(ValueError, match="w_hat_int8 length"):
        encode_z3hv2_section(
            hyperprior_weights_int8=section_meta["weights_int8"],
            w_hat_int8=b"\x00" * 5,
            residual_int8=section_meta["residual_int8"],
            latent_min=section_meta["latent_min"],
            latent_scale=section_meta["latent_scale"],
            hyper_dim=8,
            int8_w_scale=1.0,
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
            factorized_half_range=16.0,
        )


def test_z3hv2_encode_refuses_residual_length_mismatch():
    section_meta = _build_synthetic_z3hv2_section()[1]
    with pytest.raises(ValueError, match="residual_int8 length"):
        encode_z3hv2_section(
            hyperprior_weights_int8=section_meta["weights_int8"],
            w_hat_int8=section_meta["w_hat_int8"],
            residual_int8=b"\x00" * 5,
            latent_min=section_meta["latent_min"],
            latent_scale=section_meta["latent_scale"],
            hyper_dim=8,
            int8_w_scale=1.0,
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
            factorized_half_range=16.0,
        )


def test_z3hv2_encode_refuses_bad_hyper_dim():
    section_meta = _build_synthetic_z3hv2_section()[1]
    with pytest.raises(ValueError, match="hyper_dim must be"):
        encode_z3hv2_section(
            hyperprior_weights_int8=section_meta["weights_int8"],
            w_hat_int8=section_meta["w_hat_int8"],
            residual_int8=section_meta["residual_int8"],
            latent_min=section_meta["latent_min"],
            latent_scale=section_meta["latent_scale"],
            hyper_dim=0,
            int8_w_scale=1.0,
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
            factorized_half_range=16.0,
        )


# ---------------------------------------------------------------------------
# v2 payload assembly + split tests
# ---------------------------------------------------------------------------


def test_build_z3v2_payload_replaces_latent_blob():
    a1 = _build_synthetic_a1_bytes()
    section, _ = _build_synthetic_z3hv2_section()
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=section)
    # Decoder section preserved verbatim.
    assert payload[:A1_DECODER_SECTION_TOTAL] == a1[:A1_DECODER_SECTION_TOTAL]
    # Z3HV2 magic at the latent_blob slot.
    assert payload[A1_DECODER_SECTION_TOTAL : A1_DECODER_SECTION_TOTAL + 4] == Z3HV2_MAGIC
    # Sidecar preserved verbatim.
    assert payload[-607:] == a1[-607:]


def test_build_z3v2_payload_refuses_short_a1():
    a1_short = b"X" * 1000  # too short
    section, _ = _build_synthetic_z3hv2_section()
    with pytest.raises(ValueError, match="a1_bytes too short"):
        build_z3v2_payload_bytes(a1_bytes=a1_short, z3hv2_section=section)


def test_build_z3v2_payload_refuses_section_without_magic():
    a1 = _build_synthetic_a1_bytes()
    bad_section = b"NOT_Z3V2_MAGIC" + b"\x00" * 100
    with pytest.raises(ValueError, match="does not start with magic"):
        build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=bad_section)


def test_split_z3v2_payload_roundtrip():
    a1 = _build_synthetic_a1_bytes()
    section, _ = _build_synthetic_z3hv2_section()
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=section)
    decoder, z3hv2, sidecar = split_z3v2_payload_bytes(payload)
    assert decoder == a1[:A1_DECODER_SECTION_TOTAL]
    assert z3hv2 == section
    assert sidecar == a1[A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN :]


def test_split_z3v2_payload_refuses_v1_layout():
    # v1 has Z3HP1 magic at end of A1 bytes (NOT at decoder boundary).
    a1 = _build_synthetic_a1_bytes()
    v1_payload = a1 + b"Z3H1" + b"\x00" * 100
    with pytest.raises(ValueError, match="missing Z3HV2 magic"):
        split_z3v2_payload_bytes(v1_payload)


# ---------------------------------------------------------------------------
# Composition contract tests
# ---------------------------------------------------------------------------


def test_v2_contract_byte_saving_when_z3hv2_smaller_than_a1_latent_blob():
    a1 = _build_synthetic_a1_bytes()
    # Build a small section (compress-friendly all-zero input).
    section_meta = _build_synthetic_z3hv2_section()[1]
    small_residual = b"\x00" * (A1_N_PAIRS * A1_LATENT_DIM)
    small_w_hat = b"\x00" * (A1_N_PAIRS * 8)
    small_section = encode_z3hv2_section(
        hyperprior_weights_int8=section_meta["weights_int8"],
        w_hat_int8=small_w_hat,
        residual_int8=small_residual,
        latent_min=section_meta["latent_min"],
        latent_scale=section_meta["latent_scale"],
        hyper_dim=8,
        int8_w_scale=section_meta["int8_w_scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
        factorized_half_range=16.0,
    )
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=small_section)
    contract = build_z3v2_composition_archive_contract(a1, payload)
    assert isinstance(contract, Z3V2CompositionArchiveContract)
    assert contract.layout == "z3v2_latent_replacement"
    assert contract.byte_saving is True
    assert contract.byte_savings_bytes > 0
    assert contract.score_claim is False
    assert contract.promotion_eligible is False
    assert contract.ready_for_exact_eval_dispatch is False
    manifest = contract.as_manifest()
    assert manifest["z3v2_section_bytes"] < A1_LATENT_BLOB_LEN


def test_v2_contract_no_byte_saving_when_section_larger():
    a1 = _build_synthetic_a1_bytes()
    section, _ = _build_synthetic_z3hv2_section()  # high-entropy residual ~ poor compression
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=section)
    contract = build_z3v2_composition_archive_contract(a1, payload)
    # If section_bytes >= A1_LATENT_BLOB_LEN, byte_saving is False.
    if contract.z3v2_section_bytes >= A1_LATENT_BLOB_LEN:
        assert contract.byte_saving is False
        assert contract.byte_savings_bytes == 0


def test_v2_contract_refuses_wrong_decoder_section():
    a1 = _build_synthetic_a1_bytes()
    section, _ = _build_synthetic_z3hv2_section()
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=section)
    # Tamper with decoder section.
    bad_payload = b"X" * A1_DECODER_SECTION_TOTAL + payload[A1_DECODER_SECTION_TOTAL:]
    with pytest.raises(ValueError, match="decoder section diverges"):
        build_z3v2_composition_archive_contract(a1, bad_payload)


def test_v2_contract_refuses_wrong_sidecar_section():
    a1 = _build_synthetic_a1_bytes()
    section, _ = _build_synthetic_z3hv2_section()
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=section)
    # Tamper with trailing sidecar.
    bad_payload = payload[:-100] + b"X" * 100
    with pytest.raises(ValueError, match="sidecar diverges"):
        build_z3v2_composition_archive_contract(a1, bad_payload)


# ---------------------------------------------------------------------------
# inflate_v2.py dispatch tests
# ---------------------------------------------------------------------------


def test_is_v2_payload_returns_true_for_v2():
    a1 = _build_synthetic_a1_bytes()
    section, _ = _build_synthetic_z3hv2_section()
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=section)
    assert is_v2_payload(payload) is True


def test_is_v2_payload_returns_false_for_a1_bytes():
    a1 = _build_synthetic_a1_bytes()
    assert is_v2_payload(a1) is False


def test_is_v2_payload_returns_false_for_v1_sidecar():
    a1 = _build_synthetic_a1_bytes()
    v1_payload = a1 + b"Z3H1" + b"\x00" * 50
    assert is_v2_payload(v1_payload) is False


def test_is_v2_payload_returns_false_for_short_bytes():
    assert is_v2_payload(b"") is False
    assert is_v2_payload(b"\x00" * 100) is False


def test_select_inflate_device_default_is_cpu_when_no_cuda(monkeypatch):
    monkeypatch.delenv("PACT_INFLATE_DEVICE", raising=False)
    # On a CUDA-less environment, select_inflate_device returns cpu.
    if not torch.cuda.is_available():
        assert select_inflate_device().type == "cpu"


def test_select_inflate_device_refuses_mps(monkeypatch):
    monkeypatch.setenv("PACT_INFLATE_DEVICE", "mps")
    with pytest.raises(RuntimeError, match="MPS is noise"):
        select_inflate_device()


def test_select_inflate_device_explicit_cpu(monkeypatch):
    monkeypatch.setenv("PACT_INFLATE_DEVICE", "cpu")
    assert select_inflate_device().type == "cpu"


# ---------------------------------------------------------------------------
# End-to-end reconstruction tests
# ---------------------------------------------------------------------------


def test_reconstruct_a1_latents_from_v2_payload_returns_correct_shape():
    a1 = _build_synthetic_a1_bytes()
    section, _ = _build_synthetic_z3hv2_section()
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=section)
    a1_byte_faithful, latents = reconstruct_a1_latents_from_v2_payload(payload)
    assert latents.shape == (A1_N_PAIRS, A1_LATENT_DIM)
    assert latents.dtype == torch.float32
    # a1_byte_faithful preserves decoder + sidecar sections.
    assert a1_byte_faithful[:A1_DECODER_SECTION_TOTAL] == a1[:A1_DECODER_SECTION_TOTAL]
    assert a1_byte_faithful[-607:] == a1[-607:]
    # The latent_blob slot in a1_byte_faithful is zero-padded (the latents
    # are returned via the second tuple element directly, not in A1's LZMA
    # format).
    assert a1_byte_faithful[
        A1_DECODER_SECTION_TOTAL : A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN
    ] == b"\x00" * A1_LATENT_BLOB_LEN


def test_reconstruct_a1_latents_from_v2_payload_finite():
    a1 = _build_synthetic_a1_bytes()
    section, _ = _build_synthetic_z3hv2_section()
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=section)
    _, latents = reconstruct_a1_latents_from_v2_payload(payload)
    assert torch.isfinite(latents).all()


def test_reconstruct_a1_latents_dispatches_v2_correctly():
    a1 = _build_synthetic_a1_bytes()
    section, _ = _build_synthetic_z3hv2_section()
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=section)
    out_bytes, latents = reconstruct_a1_latents(payload)
    assert latents is not None  # v2 path returns non-None latents.
    assert latents.shape == (A1_N_PAIRS, A1_LATENT_DIM)


def test_reconstruct_a1_latents_dispatches_a1_byte_identical_correctly():
    a1 = _build_synthetic_a1_bytes()
    out_bytes, latents = reconstruct_a1_latents(a1)
    assert latents is None  # A1-byte-identical path returns None.
    assert out_bytes == a1


def test_v2_payload_bytes_smaller_than_a1_when_compressible():
    """Sanity: a v2 payload with all-zero residual is SMALLER than A1.

    This is the dispositive byte-saving test: prove the v2 grammar can
    realize the predicted savings.
    """
    a1 = _build_synthetic_a1_bytes()
    section_meta = _build_synthetic_z3hv2_section()[1]
    small_section = encode_z3hv2_section(
        hyperprior_weights_int8=section_meta["weights_int8"],
        w_hat_int8=b"\x00" * (A1_N_PAIRS * 8),
        residual_int8=b"\x00" * (A1_N_PAIRS * A1_LATENT_DIM),
        latent_min=section_meta["latent_min"],
        latent_scale=section_meta["latent_scale"],
        hyper_dim=8,
        int8_w_scale=section_meta["int8_w_scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
        factorized_half_range=16.0,
    )
    payload = build_z3v2_payload_bytes(a1_bytes=a1, z3hv2_section=small_section)
    # The payload should be SMALLER than the A1 archive (proving v2 has the
    # structural capacity to realize byte savings).
    assert len(payload) < len(a1), (
        f"v2 payload {len(payload)} bytes is NOT smaller than A1 {len(a1)} bytes"
    )
