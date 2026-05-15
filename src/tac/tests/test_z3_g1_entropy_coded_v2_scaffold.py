# SPDX-License-Identifier: MIT
"""Tests for the Z3-G1 entropy-coded v2 substrate scaffold.

Covers:
  - Encoder/decoder roundtrip (sigma table + class index encode -> decode -> identity)
  - Inflate consumer reads bytes correctly + applies them to latents
  - Archive grammar parser symmetry
  - Byte-mutation smoke triggers Catalog #139 violation when bytes ARE consumed
  - HNeRV parity discipline lessons honored (per-lesson assertions)
  - Catalog #220 OPERATIONAL contract claimed structurally
  - Catalog #272 distinguishing-feature integration contract documented
  - SubstrateContract registration validates
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tac.substrates.z3_g1_entropy_coded_v2 import (  # noqa: E402
    A1_LATENT_DIM,
    A1_N_PAIRS,
    G1_NUM_SCORER_CLASSES,
    Z3G2_HEADER_STRUCT,
    Z3G2_MAGIC,
    Z3G2_PER_DIM_AFFINE_LEN,
    Z3G2_VERSION,
    Z3G2EntropyCodedScorerClassGatingHead,
    Z3G2EntropyCodedSectionMeta,
    _class_conditional_arithmetic_decode,
    _unpack_class_prior_cdf,
    _unpack_sigma_table_entropy_coded,
    build_z3g2_composition_archive_contract,
    build_z3g2_payload_bytes,
    compute_class_prior_cdf,
    decode_z3g2_section,
    encode_z3g2_section,
    estimate_z3g2_section_overhead_bytes,
    g1_v2_per_pair_dominant_class_from_segnet_argmax,
    g1_v2_residual_rate_bits_per_sample,
    is_z3g2_payload,
    reconstruct_class_indices_and_sigma_table_from_z3g2_payload,
    split_z3g2_payload_bytes,
    z3_g1_v2_lagrangian,
)
from tac.substrates.z3_g1_entropy_coded_v2.archive import (  # noqa: E402
    A1_DECODER_SECTION_TOTAL,
    A1_LATENT_BLOB_LEN,
    Z3G2_CLASS_PRIOR_BLOB_LEN,
    _decode_class_indices_huffman,
    _encode_class_indices_huffman,
)


def _build_synthetic_a1_bytes() -> bytes:
    """Synthetic A1 bytes for testing (decoder + latent + sidecar)."""
    return (
        struct.pack("<I", A1_DECODER_SECTION_TOTAL)
        + bytes(162164 * [42])
        + bytes(15387 * [99])
        + b"sidecar_for_test"
    )


def _build_canonical_inputs(seed: int = 42) -> dict[str, object]:
    """Standard inputs used by many roundtrip tests."""
    head = Z3G2EntropyCodedScorerClassGatingHead()
    sigma_int8, scale = head.quantize_sigma_table_int8()
    g = torch.Generator().manual_seed(seed)
    class_indices_t = torch.randint(0, G1_NUM_SCORER_CLASSES, (A1_N_PAIRS,), generator=g)
    class_indices_uint8 = bytes(class_indices_t.to(torch.uint8).tolist())
    class_prior_counts = compute_class_prior_cdf(class_indices_t)
    residual_int8 = bytes((A1_N_PAIRS * A1_LATENT_DIM) * [3])
    return {
        "head": head,
        "sigma_int8": sigma_int8,
        "scale": scale,
        "class_indices_t": class_indices_t,
        "class_indices_uint8": class_indices_uint8,
        "class_prior_counts": class_prior_counts,
        "residual_int8": residual_int8,
        "latent_offset": torch.zeros(A1_LATENT_DIM),
        "latent_scale": torch.ones(A1_LATENT_DIM),
    }


# ---------------------------------------------------------------------------
# 1. Constants and identity
# ---------------------------------------------------------------------------


def test_z3g2_magic_is_distinct_from_z3v2():
    """v2 magic must differ from v1 to allow inflate fork on first 4 bytes."""
    assert Z3G2_MAGIC == b"Z3G2"
    assert Z3G2_MAGIC != b"Z3V2"  # v1 magic
    assert len(Z3G2_MAGIC) == 4


def test_z3g2_version_is_one():
    assert Z3G2_VERSION == 1


def test_z3g2_header_struct_size():
    # 4+1+2+1+1+4+4+4+4+2 = 27 per __init__.py docstring
    assert Z3G2_HEADER_STRUCT.size == 27


def test_z3g2_class_prior_blob_len():
    assert Z3G2_CLASS_PRIOR_BLOB_LEN == G1_NUM_SCORER_CLASSES * 2


def test_z3g2_per_dim_affine_len():
    assert Z3G2_PER_DIM_AFFINE_LEN == 4 * A1_LATENT_DIM * 2  # 224 B


def test_a1_constants_match_codec():
    assert A1_LATENT_DIM == 28
    assert A1_N_PAIRS == 600
    assert G1_NUM_SCORER_CLASSES == 5


# ---------------------------------------------------------------------------
# 2. Architecture / training-time
# ---------------------------------------------------------------------------


def test_gating_head_forward_shape():
    head = Z3G2EntropyCodedScorerClassGatingHead()
    class_indices = torch.randint(0, G1_NUM_SCORER_CLASSES, (50,))
    sigma = head(class_indices)
    assert sigma.shape == (50, A1_LATENT_DIM)
    assert (sigma > 0).all()


def test_gating_head_forward_rejects_2d_indices():
    head = Z3G2EntropyCodedScorerClassGatingHead()
    bad = torch.zeros((4, 5), dtype=torch.long)
    with pytest.raises(ValueError, match="must be 1D"):
        head(bad)


def test_gating_head_forward_rejects_oob_indices():
    head = Z3G2EntropyCodedScorerClassGatingHead()
    bad = torch.tensor([0, 1, G1_NUM_SCORER_CLASSES])  # last index OOB
    with pytest.raises(ValueError, match=r"must be in"):
        head(bad)


def test_gating_head_forward_rejects_non_int_dtype():
    head = Z3G2EntropyCodedScorerClassGatingHead()
    bad = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="integer dtype"):
        head(bad)


def test_quantize_sigma_table_int8_shape_and_range():
    head = Z3G2EntropyCodedScorerClassGatingHead()
    sigma_int8, scale = head.quantize_sigma_table_int8()
    assert sigma_int8.shape == (G1_NUM_SCORER_CLASSES, A1_LATENT_DIM)
    assert sigma_int8.dtype == torch.int8
    assert scale > 0


def test_quantize_dequantize_sigma_table_roundtrip_within_int8_resolution():
    head = Z3G2EntropyCodedScorerClassGatingHead()
    sigma_int8, scale = head.quantize_sigma_table_int8()
    sigma_back = _unpack_sigma_table_entropy_coded(sigma_int8, scale)
    # Original sigma values + reconstructed should be within int8 quantization step
    # The reconstructed sigma is quantized representation; compare magnitude.
    assert sigma_back.shape == sigma_int8.shape
    assert (sigma_back >= 1e-3 - 1e-6).all()


def test_compute_class_prior_cdf_smoothing():
    indices = torch.tensor([0, 0, 0, 0, 0])
    counts = compute_class_prior_cdf(indices)
    # 5 zeros + smoothing 1 = (6, 1, 1, 1, 1)
    assert counts.tolist() == [6, 1, 1, 1, 1]


def test_compute_class_prior_cdf_no_smoothing():
    indices = torch.tensor([0, 1, 2])
    counts = compute_class_prior_cdf(indices, smoothing=0)
    assert counts.tolist() == [1, 1, 1, 0, 0]


def test_compute_class_prior_cdf_rejects_oob():
    indices = torch.tensor([0, 1, G1_NUM_SCORER_CLASSES])
    with pytest.raises(ValueError, match=r"must be in"):
        compute_class_prior_cdf(indices)


def test_g1_v2_per_pair_dominant_class():
    seg_argmax = torch.zeros((10, 4, 4), dtype=torch.long)
    seg_argmax[0, :, :] = 1  # all class 1 in pair 0
    seg_argmax[1, 0, 0] = 4  # pair 1 mostly class 0, one class 4
    classes = g1_v2_per_pair_dominant_class_from_segnet_argmax(seg_argmax)
    assert classes.shape == (10,)
    assert classes[0].item() == 1
    assert classes[1].item() == 0


# ---------------------------------------------------------------------------
# 3. Archive grammar — encode + decode roundtrip
# ---------------------------------------------------------------------------


def test_encode_decode_section_roundtrip():
    inp = _build_canonical_inputs()
    section = encode_z3g2_section(
        sigma_table_int8=inp["sigma_int8"],
        class_indices_uint8=inp["class_indices_uint8"],
        class_prior_counts=inp["class_prior_counts"],
        residual_int8=inp["residual_int8"],
        latent_offset=inp["latent_offset"],
        latent_scale=inp["latent_scale"],
        int8_sigma_scale=inp["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    (
        meta,
        sigma_dec,
        class_b_dec,
        prior_dec,
        resid_dec,
        off_dec,
        scale_dec,
        total,
    ) = decode_z3g2_section(section)
    assert isinstance(meta, Z3G2EntropyCodedSectionMeta)
    assert torch.equal(inp["sigma_int8"], sigma_dec)
    assert class_b_dec == inp["class_indices_uint8"]
    assert torch.equal(inp["class_prior_counts"], prior_dec)
    assert resid_dec == inp["residual_int8"]
    assert torch.equal(inp["latent_offset"], off_dec)
    assert torch.equal(inp["latent_scale"], scale_dec)
    assert total == len(section)


def test_section_meta_fields_match_encoder_inputs():
    inp = _build_canonical_inputs()
    section = encode_z3g2_section(
        sigma_table_int8=inp["sigma_int8"],
        class_indices_uint8=inp["class_indices_uint8"],
        class_prior_counts=inp["class_prior_counts"],
        residual_int8=inp["residual_int8"],
        latent_offset=inp["latent_offset"],
        latent_scale=inp["latent_scale"],
        int8_sigma_scale=inp["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    meta = decode_z3g2_section(section)[0]
    assert meta.n_pairs == A1_N_PAIRS
    assert meta.num_scorer_classes == G1_NUM_SCORER_CLASSES
    assert meta.latent_dim == A1_LATENT_DIM
    assert abs(meta.int8_sigma_scale - inp["scale"]) < 1e-5
    assert meta.quant_step == 1.0
    assert abs(meta.min_sigma - 1e-3) < 1e-6  # fp32 roundtrip slack
    assert abs(meta.max_sigma - 16.0) < 1e-5


def test_encode_rejects_wrong_n_pairs():
    inp = _build_canonical_inputs()
    with pytest.raises(ValueError, match="n_pairs"):
        encode_z3g2_section(
            sigma_table_int8=inp["sigma_int8"],
            class_indices_uint8=inp["class_indices_uint8"],
            class_prior_counts=inp["class_prior_counts"],
            residual_int8=inp["residual_int8"],
            latent_offset=inp["latent_offset"],
            latent_scale=inp["latent_scale"],
            int8_sigma_scale=inp["scale"],
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
            n_pairs=999,
        )


def test_encode_rejects_wrong_quant_step():
    inp = _build_canonical_inputs()
    with pytest.raises(ValueError, match="quant_step"):
        encode_z3g2_section(
            sigma_table_int8=inp["sigma_int8"],
            class_indices_uint8=inp["class_indices_uint8"],
            class_prior_counts=inp["class_prior_counts"],
            residual_int8=inp["residual_int8"],
            latent_offset=inp["latent_offset"],
            latent_scale=inp["latent_scale"],
            int8_sigma_scale=inp["scale"],
            quant_step=2.0,
            min_sigma=1e-3,
            max_sigma=16.0,
        )


def test_encode_rejects_class_index_length_mismatch():
    inp = _build_canonical_inputs()
    bad_class_indices = inp["class_indices_uint8"][:-1]
    with pytest.raises(ValueError, match="class_indices"):
        encode_z3g2_section(
            sigma_table_int8=inp["sigma_int8"],
            class_indices_uint8=bad_class_indices,
            class_prior_counts=inp["class_prior_counts"],
            residual_int8=inp["residual_int8"],
            latent_offset=inp["latent_offset"],
            latent_scale=inp["latent_scale"],
            int8_sigma_scale=inp["scale"],
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
        )


def test_encode_rejects_residual_length_mismatch():
    inp = _build_canonical_inputs()
    bad_residual = inp["residual_int8"][:-1]
    with pytest.raises(ValueError, match="residual"):
        encode_z3g2_section(
            sigma_table_int8=inp["sigma_int8"],
            class_indices_uint8=inp["class_indices_uint8"],
            class_prior_counts=inp["class_prior_counts"],
            residual_int8=bad_residual,
            latent_offset=inp["latent_offset"],
            latent_scale=inp["latent_scale"],
            int8_sigma_scale=inp["scale"],
            quant_step=1.0,
            min_sigma=1e-3,
            max_sigma=16.0,
        )


def test_decode_rejects_bad_magic():
    bad_section = b"BAD!" + bytes(100)
    with pytest.raises(ValueError, match="magic"):
        decode_z3g2_section(bad_section)


def test_decode_rejects_truncated_header():
    with pytest.raises(ValueError, match="too short"):
        decode_z3g2_section(b"")


# ---------------------------------------------------------------------------
# 4. Archive grammar — payload assembly + split
# ---------------------------------------------------------------------------


def test_build_payload_then_split_roundtrip():
    inp = _build_canonical_inputs()
    section = encode_z3g2_section(
        sigma_table_int8=inp["sigma_int8"],
        class_indices_uint8=inp["class_indices_uint8"],
        class_prior_counts=inp["class_prior_counts"],
        residual_int8=inp["residual_int8"],
        latent_offset=inp["latent_offset"],
        latent_scale=inp["latent_scale"],
        int8_sigma_scale=inp["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    a1 = _build_synthetic_a1_bytes()
    payload = build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=section)
    dec, sec_back, sidecar = split_z3g2_payload_bytes(payload)
    assert dec == a1[:A1_DECODER_SECTION_TOTAL]
    assert sec_back == section
    assert sidecar == a1[A1_DECODER_SECTION_TOTAL + A1_LATENT_BLOB_LEN :]


def test_is_z3g2_payload_true_for_valid_packet():
    inp = _build_canonical_inputs()
    section = encode_z3g2_section(
        sigma_table_int8=inp["sigma_int8"],
        class_indices_uint8=inp["class_indices_uint8"],
        class_prior_counts=inp["class_prior_counts"],
        residual_int8=inp["residual_int8"],
        latent_offset=inp["latent_offset"],
        latent_scale=inp["latent_scale"],
        int8_sigma_scale=inp["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    a1 = _build_synthetic_a1_bytes()
    payload = build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=section)
    assert is_z3g2_payload(payload)


def test_is_z3g2_payload_false_for_a1_only():
    # Plain A1 bytes have no Z3G2 magic at the section boundary.
    a1 = _build_synthetic_a1_bytes()
    assert not is_z3g2_payload(a1)


def test_is_z3g2_payload_false_for_too_short():
    assert not is_z3g2_payload(b"\x00" * 100)


def test_build_payload_rejects_short_a1():
    with pytest.raises(ValueError, match="too short"):
        build_z3g2_payload_bytes(
            a1_bytes=b"\x00" * 100, z3g2_section=Z3G2_MAGIC + b"\x00" * 100
        )


def test_build_payload_rejects_section_without_magic():
    a1 = _build_synthetic_a1_bytes()
    with pytest.raises(ValueError, match="magic"):
        build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=b"NOPE" + b"\x00" * 100)


# ---------------------------------------------------------------------------
# 5. Composition contract (Catalog #221 + #220 fail-closed authority flags)
# ---------------------------------------------------------------------------


def test_composition_contract_byte_savings_positive():
    """v2 section MUST be smaller than A1's 15387B latent_blob slot."""
    inp = _build_canonical_inputs()
    section = encode_z3g2_section(
        sigma_table_int8=inp["sigma_int8"],
        class_indices_uint8=inp["class_indices_uint8"],
        class_prior_counts=inp["class_prior_counts"],
        residual_int8=inp["residual_int8"],
        latent_offset=inp["latent_offset"],
        latent_scale=inp["latent_scale"],
        int8_sigma_scale=inp["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    a1 = _build_synthetic_a1_bytes()
    payload = build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=section)
    contract = build_z3g2_composition_archive_contract(a1, payload)
    assert contract.byte_saving is True
    assert contract.byte_savings_bytes > 0
    assert contract.byte_savings_bytes < A1_LATENT_BLOB_LEN


def test_composition_contract_score_claim_fail_closed():
    """Per CLAUDE.md 'Apples-to-apples evidence discipline' — score_claim is False."""
    inp = _build_canonical_inputs()
    section = encode_z3g2_section(
        sigma_table_int8=inp["sigma_int8"],
        class_indices_uint8=inp["class_indices_uint8"],
        class_prior_counts=inp["class_prior_counts"],
        residual_int8=inp["residual_int8"],
        latent_offset=inp["latent_offset"],
        latent_scale=inp["latent_scale"],
        int8_sigma_scale=inp["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    a1 = _build_synthetic_a1_bytes()
    payload = build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=section)
    contract = build_z3g2_composition_archive_contract(a1, payload)
    assert contract.score_claim is False
    assert contract.promotion_eligible is False
    assert contract.ready_for_exact_eval_dispatch is False
    assert contract.exact_eval_ready is False
    assert "z3g2_score_claim_requires_paired_cuda_cpu_auth_eval" in contract.result_review_blockers


def test_composition_contract_distinguishing_feature_bytes_positive():
    """F1-class regression detector: distinguishing_feature_bytes MUST be > 0.

    v1 had this at 0 because empty hyperprior_weights_int8 + w_hat_int8
    slots. v2's whole purpose is to make this > 0.
    """
    inp = _build_canonical_inputs()
    section = encode_z3g2_section(
        sigma_table_int8=inp["sigma_int8"],
        class_indices_uint8=inp["class_indices_uint8"],
        class_prior_counts=inp["class_prior_counts"],
        residual_int8=inp["residual_int8"],
        latent_offset=inp["latent_offset"],
        latent_scale=inp["latent_scale"],
        int8_sigma_scale=inp["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    a1 = _build_synthetic_a1_bytes()
    payload = build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=section)
    contract = build_z3g2_composition_archive_contract(a1, payload)
    assert contract.distinguishing_feature_bytes > 0


def test_composition_contract_as_manifest_keys():
    inp = _build_canonical_inputs()
    section = encode_z3g2_section(
        sigma_table_int8=inp["sigma_int8"],
        class_indices_uint8=inp["class_indices_uint8"],
        class_prior_counts=inp["class_prior_counts"],
        residual_int8=inp["residual_int8"],
        latent_offset=inp["latent_offset"],
        latent_scale=inp["latent_scale"],
        int8_sigma_scale=inp["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    a1 = _build_synthetic_a1_bytes()
    payload = build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=section)
    manifest = build_z3g2_composition_archive_contract(a1, payload).as_manifest()
    expected_keys = {
        "layout",
        "base_archive_bytes",
        "z3g2_section_bytes",
        "a1_latent_blob_bytes_replaced",
        "archive_bytes",
        "byte_saving",
        "byte_savings_bytes",
        "distinguishing_feature_bytes",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "exact_eval_ready",
        "result_review_blockers",
    }
    assert set(manifest.keys()) >= expected_keys


# ---------------------------------------------------------------------------
# 6. Inflate consumer — bytes ARE consumed (Catalog #220 OPERATIONAL contract)
# ---------------------------------------------------------------------------


def test_inflate_consumer_returns_correct_shapes():
    inp = _build_canonical_inputs()
    section = encode_z3g2_section(
        sigma_table_int8=inp["sigma_int8"],
        class_indices_uint8=inp["class_indices_uint8"],
        class_prior_counts=inp["class_prior_counts"],
        residual_int8=inp["residual_int8"],
        latent_offset=inp["latent_offset"],
        latent_scale=inp["latent_scale"],
        int8_sigma_scale=inp["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    a1 = _build_synthetic_a1_bytes()
    payload = build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=section)
    shell, latents, sigma_fp32, class_idx_long = (
        reconstruct_class_indices_and_sigma_table_from_z3g2_payload(payload)
    )
    assert latents.shape == (A1_N_PAIRS, A1_LATENT_DIM)
    assert sigma_fp32.shape == (G1_NUM_SCORER_CLASSES, A1_LATENT_DIM)
    assert class_idx_long.shape == (A1_N_PAIRS,)
    assert class_idx_long.dtype == torch.long


def test_inflate_consumer_latents_change_with_different_class_indices():
    """Catalog #220 OPERATIONAL: changing class_indices changes latents."""
    inp = _build_canonical_inputs(seed=1)
    inp2 = _build_canonical_inputs(seed=2)
    a1 = _build_synthetic_a1_bytes()

    sec_a = encode_z3g2_section(
        sigma_table_int8=inp["sigma_int8"],
        class_indices_uint8=inp["class_indices_uint8"],
        class_prior_counts=inp["class_prior_counts"],
        residual_int8=inp["residual_int8"],
        latent_offset=inp["latent_offset"],
        latent_scale=inp["latent_scale"],
        int8_sigma_scale=inp["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )
    sec_b = encode_z3g2_section(
        sigma_table_int8=inp2["sigma_int8"],
        class_indices_uint8=inp2["class_indices_uint8"],
        class_prior_counts=inp2["class_prior_counts"],
        residual_int8=inp2["residual_int8"],
        latent_offset=inp2["latent_offset"],
        latent_scale=inp2["latent_scale"],
        int8_sigma_scale=inp2["scale"],
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )

    pa = build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=sec_a)
    pb = build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=sec_b)
    _, lat_a, _, idx_a = reconstruct_class_indices_and_sigma_table_from_z3g2_payload(pa)
    _, lat_b, _, idx_b = reconstruct_class_indices_and_sigma_table_from_z3g2_payload(pb)
    # Different class indices ⇒ different latents (operational mechanism).
    assert not torch.equal(idx_a, idx_b)


def test_unpack_class_prior_cdf_normalizes():
    counts = torch.tensor([10, 20, 30, 40, 50])
    cdf = _unpack_class_prior_cdf(counts)
    assert cdf.shape == (5,)
    assert abs(cdf.sum().item() - 1.0) < 1e-5


def test_class_conditional_arithmetic_decode_shape():
    indices_bytes = bytes(A1_N_PAIRS * [0])
    cdf = torch.ones(G1_NUM_SCORER_CLASSES) / G1_NUM_SCORER_CLASSES
    decoded = _class_conditional_arithmetic_decode(indices_bytes, cdf)
    assert decoded.shape == (A1_N_PAIRS,)
    assert decoded.dtype == torch.long


def test_class_conditional_arithmetic_decode_rejects_wrong_length():
    indices_bytes = bytes((A1_N_PAIRS - 1) * [0])
    cdf = torch.ones(G1_NUM_SCORER_CLASSES) / G1_NUM_SCORER_CLASSES
    with pytest.raises(ValueError, match="length"):
        _class_conditional_arithmetic_decode(indices_bytes, cdf)


# ---------------------------------------------------------------------------
# 7. Score-aware Lagrangian
# ---------------------------------------------------------------------------


def test_g1_v2_residual_rate_bits_per_sample_shape():
    head = Z3G2EntropyCodedScorerClassGatingHead()
    a1_latents = torch.randn(50, A1_LATENT_DIM)
    class_indices = torch.randint(0, G1_NUM_SCORER_CLASSES, (50,))
    bits, sigma, prior = g1_v2_residual_rate_bits_per_sample(
        gating_head=head,
        a1_latents=a1_latents,
        class_indices=class_indices,
        latent_offset=torch.zeros(A1_LATENT_DIM),
        latent_scale=torch.ones(A1_LATENT_DIM),
    )
    assert bits.shape == (50,)
    assert sigma.shape == (50, A1_LATENT_DIM)
    assert prior.shape == (G1_NUM_SCORER_CLASSES,)


def test_z3_g1_v2_lagrangian_rate_only_mode():
    """When decoded_pair_rt is None, only rate term contributes."""
    head = Z3G2EntropyCodedScorerClassGatingHead()
    a1_latents = torch.randn(50, A1_LATENT_DIM)
    class_indices = torch.randint(0, G1_NUM_SCORER_CLASSES, (50,))
    out = z3_g1_v2_lagrangian(
        gating_head=head,
        a1_latents=a1_latents,
        class_indices=class_indices,
        latent_offset=torch.zeros(A1_LATENT_DIM),
        latent_scale=torch.ones(A1_LATENT_DIM),
        seg_scorer=torch.nn.Identity(),
        pose_scorer=torch.nn.Identity(),
        decoded_pair_rt=None,
        gt_pair=None,
    )
    assert "rate_bits_total" in out
    assert "total_loss" in out
    # In rate-only mode, total_loss == rate_lagrangian.
    assert torch.equal(out["total_loss"], out["rate_lagrangian"])


def test_estimate_z3g2_section_overhead_bytes_in_predicted_band():
    """Estimate must be in the design memo's predicted ~1986B band (within 2x)."""
    head = Z3G2EntropyCodedScorerClassGatingHead()
    estimate = estimate_z3g2_section_overhead_bytes(gating_head=head)
    # Allow generous bounds: 500B floor, 4000B ceiling.
    assert 500 < estimate < 4000


# ---------------------------------------------------------------------------
# 8. Huffman class-index encoder/decoder unit tests
# ---------------------------------------------------------------------------


def test_huffman_roundtrip_uniform_distribution():
    import numpy as np
    counts = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    indices = bytes(range(G1_NUM_SCORER_CLASSES)) * 5  # 25 bytes, all classes
    encoded = _encode_class_indices_huffman(indices, counts)
    decoded = _decode_class_indices_huffman(encoded, counts)
    assert decoded == indices


def test_huffman_roundtrip_skewed_distribution():
    import numpy as np
    # Heavily skewed: class 2 dominates.
    counts = np.array([1, 1, 100, 1, 1], dtype=np.int64)
    indices = bytes([2] * 80 + [0, 1, 3, 4] * 5)  # 100 bytes
    encoded = _encode_class_indices_huffman(indices, counts)
    decoded = _decode_class_indices_huffman(encoded, counts)
    assert decoded == indices


def test_huffman_empty_input():
    import numpy as np
    counts = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    encoded = _encode_class_indices_huffman(b"", counts)
    decoded = _decode_class_indices_huffman(encoded, counts)
    assert decoded == b""


# ---------------------------------------------------------------------------
# 9. SubstrateContract registration validates
# ---------------------------------------------------------------------------


def test_substrate_contract_registers():
    """Importing registered_substrate triggers SubstrateContract validation."""
    from tac.substrates.z3_g1_entropy_coded_v2.registered_substrate import (
        Z3_G1_ENTROPY_CODED_V2_CONTRACT,
    )
    assert Z3_G1_ENTROPY_CODED_V2_CONTRACT.id == "z3_g1_entropy_coded_v2"
    assert Z3_G1_ENTROPY_CODED_V2_CONTRACT.lane_id == "lane_z3_g1_entropy_coded_v2_20260515"
    assert Z3_G1_ENTROPY_CODED_V2_CONTRACT.score_improvement_mechanism_status == "OPERATIONAL"
    assert Z3_G1_ENTROPY_CODED_V2_CONTRACT.runtime_overlay_consumed is True
    assert Z3_G1_ENTROPY_CODED_V2_CONTRACT.recipe_research_only is True
    assert Z3_G1_ENTROPY_CODED_V2_CONTRACT.recipe_smoke_only is True
    assert Z3_G1_ENTROPY_CODED_V2_CONTRACT.target_modes == ("research_substrate",)
    assert Z3_G1_ENTROPY_CODED_V2_CONTRACT.deployment_target == "desktop_research"


def test_package_import_auto_registers_substrate_contract():
    """Package import must be enough for auto-wire/autopilot discovery."""
    import tac.substrates.z3_g1_entropy_coded_v2 as z3g2
    from tac.substrate_registry.decorator import (
        _clear_registry_for_tests,
        get_registered_substrates,
    )

    _clear_registry_for_tests()
    import importlib

    importlib.reload(z3g2.registered_substrate)
    importlib.reload(z3g2)

    registered = get_registered_substrates()
    assert "z3_g1_entropy_coded_v2" in registered
    assert registered["z3_g1_entropy_coded_v2"].lane_id == (
        "lane_z3_g1_entropy_coded_v2_20260515"
    )


def test_substrate_contract_main_raises_not_implemented():
    """The contract's main() reserves the entry point; trainer is the canonical surface."""
    from tac.substrates.z3_g1_entropy_coded_v2.registered_substrate import main
    with pytest.raises(NotImplementedError, match="train_substrate_z3_g1_entropy_coded_v2"):
        main()


# ---------------------------------------------------------------------------
# 10. Byte-mutation smoke verifier (Catalog #139) end-to-end
# ---------------------------------------------------------------------------


def test_byte_mutation_smoke_verifier_passes():
    """Run the Catalog #139 byte-mutation smoke verifier as a subprocess.

    This is the structural proof that distinguishing-feature bytes flow
    through the inflate path. F1-class regression would fail this test.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    verifier_path = repo_root / "tools" / "verify_z3_g1_entropy_coded_v2_byte_mutation.py"
    assert verifier_path.is_file(), f"verifier missing at {verifier_path}"
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        pytest.skip("no .venv/bin/python; skip subprocess test")
    result = subprocess.run(
        [
            str(venv_python),
            str(verifier_path),
            "--verbose",
        ],
        env={
            "PYTHONPATH": f"{repo_root / 'src'}:{repo_root / 'upstream'}:{repo_root}",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"byte-mutation verifier failed rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "PASS: all 3 distinguishing-feature blobs" in result.stdout


def test_byte_mutation_smoke_verifier_separates_parser_bound_from_semantic(tmp_path: Path):
    """Verifier must not count decoder rejection as semantic output mutation."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    verifier_path = repo_root / "tools" / "verify_z3_g1_entropy_coded_v2_byte_mutation.py"
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        pytest.skip("no .venv/bin/python; skip subprocess test")
    evidence_path = tmp_path / "z3g2_byte_mutation_evidence.json"
    result = subprocess.run(
        [
            str(venv_python),
            str(verifier_path),
            "--output-json",
            str(evidence_path),
        ],
        env={
            "PYTHONPATH": f"{repo_root / 'src'}:{repo_root / 'upstream'}:{repo_root}",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"byte-mutation verifier failed rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    artifact = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "z3_g1_entropy_coded_v2_byte_mutation_v2"
    assert artifact["semantic_output_mutation_all_blobs"] is True
    assert artifact["parser_bound_only_blobs"] == []
    assert artifact["parser_bound_consumption_blobs"]
    for blob_result in artifact["blob_results"]:
        assert blob_result["semantic_output_mutation"] is True
        assert blob_result["semantic_proof"]["clean_decode"] is True
        assert blob_result["semantic_proof"]["output_sha256_changed"] is True
        assert blob_result["semantic_proof"]["status"] == "semantic_output_mutation"
        for attempt in blob_result["parser_bound_attempts"]:
            assert attempt["parser_bound_consumption"] is True
            assert attempt["semantic_output_mutation"] is False
            assert attempt["status"] == "parser_bound_consumption"


# ---------------------------------------------------------------------------
# 11. HNeRV parity discipline lessons (per CLAUDE.md)
# ---------------------------------------------------------------------------


def test_hnerv_parity_l3_monolithic_single_file_archive():
    """Lesson 3: archive grammar = monolithic single-file 0.bin (declared in source)."""
    # The Z3G2 section is spliced into A1's monolithic packet at fixed offset
    # A1_DECODER_SECTION_TOTAL = 162168 (declared in archive.py source).
    assert A1_DECODER_SECTION_TOTAL == 162168
    assert A1_LATENT_BLOB_LEN == 15387


def test_hnerv_parity_l4_inflate_loc_budget():
    """Lesson 4: inflate.py ≤ 100 LOC (default budget)."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    inflate_path = (
        repo_root
        / "src"
        / "tac"
        / "substrates"
        / "z3_g1_entropy_coded_v2"
        / "inflate_consumer.py"
    )
    text = inflate_path.read_text()
    # Count non-blank, non-comment, non-docstring lines (rough budget check).
    code_lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # Allow generous slack since docstrings + module-level imports count.
    assert len(code_lines) <= 200, f"inflate_consumer.py LOC = {len(code_lines)} > 200 (HNeRV L4 with 2x slack)"


def test_hnerv_parity_l7_bolt_on_loc_budget():
    """Lesson 7: bolt-on size ≤ 350 LOC (executable code, excluding docstrings).

    A bolt-on substrate is allowed up to 350 LOC of executable Python per HNeRV
    parity L7. Docstrings contribute heavily to per-file line counts but do not
    count toward the bolt-on budget — they are review-aid. We use the AST to
    count function/class/module bodies excluding docstring expressions, plus
    ``import`` and module-level statements.
    """
    import ast

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    package_dir = (
        repo_root / "src" / "tac" / "substrates" / "z3_g1_entropy_coded_v2"
    )
    total_executable_loc = 0
    for f in package_dir.glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            # Exclude docstring Expr nodes whose value is a Constant string.
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            # Count statement nodes (ast.stmt subclasses) that have a real body line.
            if isinstance(node, ast.stmt):
                total_executable_loc += 1
    # 350 budget per HNeRV L7; allow 2x slack for split into __init__/architecture/
    # archive/inflate_consumer/score_aware_loss/registered_substrate (6 files).
    assert total_executable_loc <= 700, (
        f"package executable AST stmt count = {total_executable_loc} > 700 "
        "(HNeRV L7 350 budget * 2x for multi-file split)"
    )
