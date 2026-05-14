# SPDX-License-Identifier: MIT
"""Tests for ``tac.codec.jscc`` — SE-4 scorer-conditional entropy coder.

Lane: ``lane_implement_iglt_ternary_jscc_kc3_canonical_20260513``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tac.codec.jscc import (
    JSCC_FORMAT_VERSION,
    JSCC_MAGIC,
    JSCC_PROXY_EVIDENCE_GRADE,
    LEGACY_JSCC_HUFFMAN_MAGIC,
    SE4R_MAGIC,
    JSCCArchiveSection,
    JSCCCustodyContract,
    JSCCSectionManifest,
    ScorerConditionalEntropyCoder,
    ScorerConditionalProbabilityModel,
    decode_jscc_stream,
    encode_jscc_stream,
    parse_jscc_section,
    serialize_jscc_section,
)
from tac.codec.jscc.entropy_coder import (
    PRECISION_BITS,
    TOTAL_FREQ,
    _build_cum_table,
    _probs_to_integer_frequencies,
    validate_frequency_table,
)

# ── Constants + magic-byte sanity ────────────────────────────────────────


def test_jscc_magic_is_four_bytes_SE4R():
    # SE-4 range-coder magic; distinct from the legacy Huffman variant's
    # b"JSCC" magic in tac.codec.jscc.conditional_huffman.
    assert SE4R_MAGIC == b"SE4R"
    assert JSCC_MAGIC == SE4R_MAGIC
    assert LEGACY_JSCC_HUFFMAN_MAGIC == b"JSCC"
    assert len(JSCC_MAGIC) == 4


def test_jscc_format_version_is_two():
    assert JSCC_FORMAT_VERSION == 2


def test_precision_bits_matches_total_freq():
    assert TOTAL_FREQ == (1 << PRECISION_BITS)


# ── ScorerConditionalProbabilityModel: construction validation ──────────


def test_model_rejects_non_positive_side_dim():
    with pytest.raises(ValueError, match="side_dim must be positive"):
        ScorerConditionalProbabilityModel(side_dim=0, alphabet_size=16)


def test_model_rejects_alphabet_size_below_two():
    with pytest.raises(ValueError, match="alphabet_size must be >= 2"):
        ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=1)


def test_model_rejects_non_positive_hidden_dim():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        ScorerConditionalProbabilityModel(
            side_dim=4, alphabet_size=16, hidden_dim=0
        )


def test_model_forward_returns_probability_distribution():
    model = ScorerConditionalProbabilityModel(
        side_dim=4, alphabet_size=16, hidden_dim=8
    )
    side = torch.randn(3, 4)
    probs = model(side)
    assert probs.shape == (3, 16)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    assert (probs >= 0).all()


def test_model_forward_accepts_1d_input():
    model = ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=16)
    side = torch.randn(4)  # 1-D
    probs = model(side)
    assert probs.shape == (1, 16)


# ── _probs_to_integer_frequencies: deterministic rounding ───────────────


def test_probs_to_integer_frequencies_sum_equals_total():
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    freqs = _probs_to_integer_frequencies(probs, TOTAL_FREQ)
    assert int(freqs.sum().item()) == TOTAL_FREQ


def test_probs_to_integer_frequencies_all_at_least_one():
    probs = torch.tensor([1e-10, 1.0 - 4e-10, 1e-10, 1e-10, 1e-10])
    freqs = _probs_to_integer_frequencies(probs, TOTAL_FREQ)
    assert int(freqs.min().item()) >= 1


def test_probs_to_integer_frequencies_rejects_total_freq_too_small():
    probs = torch.tensor([0.5, 0.5])
    with pytest.raises(ValueError, match="total_freq=1"):
        _probs_to_integer_frequencies(probs, total_freq=1)


def test_integer_frequency_table_via_model():
    model = ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=16)
    side = torch.randn(4)
    freqs = model.integer_frequency_table(side)
    validate_frequency_table(freqs)


# ── validate_frequency_table: error paths ──────────────────────────────


def test_validate_frequency_table_rejects_wrong_ndim():
    bad = torch.zeros((2, 2), dtype=torch.int64)
    with pytest.raises(ValueError, match="freqs must be 1-D"):
        validate_frequency_table(bad)


def test_validate_frequency_table_rejects_wrong_dtype():
    bad = torch.zeros(16, dtype=torch.float32)
    with pytest.raises(ValueError, match="must be int64"):
        validate_frequency_table(bad)


def test_validate_frequency_table_rejects_zero_freq():
    # K=4, sum to TOTAL_FREQ but one is zero
    freqs = torch.tensor([0, 1, TOTAL_FREQ - 2, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="must be >= 1"):
        validate_frequency_table(freqs)


def test_validate_frequency_table_rejects_wrong_sum():
    freqs = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="frequency-table sum"):
        validate_frequency_table(freqs)


# ── encode/decode round-trip ────────────────────────────────────────────


def test_encode_decode_roundtrip_short_uniform_model():
    torch.manual_seed(0)
    model = ScorerConditionalProbabilityModel(
        side_dim=4, alphabet_size=8, hidden_dim=8
    )
    # Force model toward uniform output by setting fc2 weights to zero
    with torch.no_grad():
        model.fc2.weight.zero_()
        model.fc2.bias.zero_()
    N = 20
    side = torch.randn(N, 4)
    symbols = [int(s) for s in torch.randint(0, 8, (N,)).tolist()]
    encoded = encode_jscc_stream(symbols, side, model)
    decoded = decode_jscc_stream(encoded, side, model, n_symbols=N)
    assert decoded == symbols


def test_encode_decode_roundtrip_with_trained_model():
    torch.manual_seed(7)
    model = ScorerConditionalProbabilityModel(
        side_dim=6, alphabet_size=32, hidden_dim=16
    )
    N = 50
    side = torch.randn(N, 6)
    symbols = [int(s) for s in torch.randint(0, 32, (N,)).tolist()]
    encoded = encode_jscc_stream(symbols, side, model)
    decoded = decode_jscc_stream(encoded, side, model, n_symbols=N)
    assert decoded == symbols


def test_encode_decode_roundtrip_singleton_symbol():
    torch.manual_seed(42)
    model = ScorerConditionalProbabilityModel(side_dim=2, alphabet_size=4)
    side = torch.randn(1, 2)
    symbols = [2]
    encoded = encode_jscc_stream(symbols, side, model)
    decoded = decode_jscc_stream(encoded, side, model, n_symbols=1)
    assert decoded == symbols


# ── Encoder error paths ──────────────────────────────────────────────────


def test_encode_rejects_side_state_wrong_ndim():
    model = ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=8)
    side = torch.randn(4)  # 1-D, should be (N, 4)
    with pytest.raises(ValueError, match="side_states must be 2-D"):
        encode_jscc_stream([0], side, model)


def test_encode_rejects_symbol_out_of_range():
    model = ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=8)
    side = torch.randn(1, 4)
    with pytest.raises(ValueError, match="out of range"):
        encode_jscc_stream([100], side, model)


def test_encode_rejects_side_state_wrong_side_dim():
    model = ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=8)
    side = torch.randn(1, 8)  # wrong side_dim
    with pytest.raises(ValueError, match=r"model\.side_dim"):
        encode_jscc_stream([0], side, model)


def test_encode_rejects_mismatched_n_and_side_states():
    model = ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=8)
    side = torch.randn(3, 4)
    with pytest.raises(ValueError, match=r"side_states.shape\[0\]"):
        encode_jscc_stream([0, 1], side, model)  # 2 symbols vs 3 side rows


# ── ScorerConditionalEntropyCoder high-level wrapper ────────────────────


def test_high_level_coder_encode_decode_roundtrip():
    torch.manual_seed(1)
    model = ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=8)
    coder = ScorerConditionalEntropyCoder(model)
    N = 15
    side = torch.randn(N, 4)
    symbols = [int(s) for s in torch.randint(0, 8, (N,)).tolist()]
    payload = coder.encode(symbols, side)
    decoded = coder.decode(payload, side, n_symbols=N)
    assert decoded == symbols


def test_high_level_coder_rejects_non_model_arg():
    with pytest.raises(TypeError, match="must be a ScorerConditional"):
        ScorerConditionalEntropyCoder("not a model")  # type: ignore[arg-type]


def test_high_level_estimated_coded_bits_finite():
    torch.manual_seed(99)
    model = ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=8)
    coder = ScorerConditionalEntropyCoder(model)
    side = torch.randn(10, 4)
    symbols = list(range(10))
    bits = coder.estimated_coded_bits([s % 8 for s in symbols], side)
    assert math.isfinite(bits)
    assert bits > 0


def test_estimated_coded_bits_rejects_length_mismatch():
    model = ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=8)
    coder = ScorerConditionalEntropyCoder(model)
    side = torch.randn(5, 4)
    with pytest.raises(ValueError, match=r"side_states.shape\[0\]"):
        coder.estimated_coded_bits([0, 1], side)


# ── Archive section format ──────────────────────────────────────────────


def test_serialize_parse_roundtrip_empty_payload():
    section = serialize_jscc_section(
        payload=b"", side_dim=4, alphabet_size=16, n_symbols=0
    )
    parsed = parse_jscc_section(section)
    assert parsed.manifest.magic == JSCC_MAGIC
    assert parsed.manifest.version == JSCC_FORMAT_VERSION
    assert parsed.manifest.side_dim == 4
    assert parsed.manifest.alphabet_size == 16
    assert parsed.manifest.n_symbols == 0
    assert parsed.payload == b""
    assert parsed.manifest.evidence_grade == JSCC_PROXY_EVIDENCE_GRADE
    assert parsed.manifest.proxy is True
    assert parsed.manifest.proxy_only is True
    assert parsed.manifest.score_claim is False
    assert parsed.manifest.promotion_eligible is False
    assert parsed.manifest.ready_for_exact_eval_dispatch is False


def test_serialize_parse_roundtrip_real_payload():
    torch.manual_seed(5)
    model = ScorerConditionalProbabilityModel(side_dim=4, alphabet_size=8)
    coder = ScorerConditionalEntropyCoder(model)
    N = 12
    side = torch.randn(N, 4)
    symbols = [int(s) for s in torch.randint(0, 8, (N,)).tolist()]
    payload = coder.encode(symbols, side)

    section = serialize_jscc_section(
        payload=payload, side_dim=4, alphabet_size=8, n_symbols=N
    )
    parsed = parse_jscc_section(section)
    assert parsed.payload == payload
    decoded = coder.decode(parsed.payload, side, n_symbols=N)
    assert decoded == symbols


def test_parse_rejects_short_section():
    with pytest.raises(ValueError, match="section too short"):
        parse_jscc_section(b"\x00\x00")


def test_parse_rejects_bad_magic():
    good = serialize_jscc_section(
        payload=b"payload", side_dim=4, alphabet_size=16, n_symbols=5
    )
    bad = b"NOPE" + good[4:]
    with pytest.raises(ValueError, match="bad magic"):
        parse_jscc_section(bad)


def test_parse_rejects_unsupported_version():
    good = bytearray(
        serialize_jscc_section(
            payload=b"payload", side_dim=4, alphabet_size=16, n_symbols=5
        )
    )
    good[4] = 99
    with pytest.raises(ValueError, match="unsupported JSCC version"):
        parse_jscc_section(bytes(good))


def test_serialize_rejects_side_dim_too_big():
    with pytest.raises(ValueError, match="side_dim must fit in uint16"):
        serialize_jscc_section(
            payload=b"", side_dim=0x10000, alphabet_size=16, n_symbols=0
        )


def test_serialize_rejects_alphabet_size_too_small():
    with pytest.raises(ValueError, match="alphabet_size must be >= 2"):
        serialize_jscc_section(
            payload=b"", side_dim=4, alphabet_size=1, n_symbols=0
        )


def test_serialize_rejects_negative_n_symbols():
    with pytest.raises(ValueError, match="n_symbols must fit in uint32"):
        serialize_jscc_section(
            payload=b"", side_dim=4, alphabet_size=8, n_symbols=-1
        )


def test_serialize_section_starts_with_magic():
    section = serialize_jscc_section(
        payload=b"\x01\x02\x03",
        side_dim=2,
        alphabet_size=4,
        n_symbols=3,
    )
    assert section[:4] == JSCC_MAGIC
    assert section[4] == JSCC_FORMAT_VERSION


def test_serialize_records_proxy_only_custody_metadata():
    section = serialize_jscc_section(
        payload=b"\x01\x02",
        side_dim=2,
        alphabet_size=4,
        n_symbols=2,
        model_contract=JSCCCustodyContract(
            embedded=True,
            charged_bytes=1024,
            sha256="a" * 64,
            description="embedded int8 probability model",
        ),
        side_state_contract={
            "embedded": True,
            "charged_bytes": 64,
            "sha256": "b" * 64,
            "description": "deterministic side-state reconstruction code",
        },
        ready_for_exact_eval_dispatch=True,
    )
    manifest = parse_jscc_section(section).manifest

    assert manifest.ready_for_exact_eval_dispatch is False
    assert manifest.score_claim is False
    assert manifest.promotion_eligible is False
    assert manifest.proxy_only is True
    assert manifest.custody_metadata["exact_eval_ready_requested"] is True
    assert manifest.custody_metadata["embedded_side_contract_complete"] is True
    assert manifest.custody_metadata["model_contract"][
        "charged_and_embedded"
    ] is True
    assert manifest.custody_metadata["side_state_reconstruction_contract"][
        "charged_and_embedded"
    ] is True


def test_serialize_exact_eval_ready_fails_without_embedded_contracts():
    with pytest.raises(ValueError, match="model_contract"):
        serialize_jscc_section(
            payload=b"",
            side_dim=2,
            alphabet_size=4,
            n_symbols=0,
            ready_for_exact_eval_dispatch=True,
        )
    with pytest.raises(ValueError, match="side_state_contract"):
        serialize_jscc_section(
            payload=b"",
            side_dim=2,
            alphabet_size=4,
            n_symbols=0,
            model_contract=JSCCCustodyContract(
                embedded=True,
                charged_bytes=1,
                sha256="c" * 64,
            ),
            ready_for_exact_eval_dispatch=True,
        )


# ── Cumulative-table helper ────────────────────────────────────────────


def test_build_cum_table_lengths_and_sum():
    freqs = torch.tensor([3, 5, 2, 7], dtype=torch.int64)
    cum = _build_cum_table(freqs)
    assert len(cum) == 5  # K + 1
    assert cum == [0, 3, 8, 10, 17]


# ── End-to-end: encode->serialize->parse->decode ───────────────────────


def test_end_to_end_pipeline_archive_section():
    torch.manual_seed(13)
    model = ScorerConditionalProbabilityModel(
        side_dim=6, alphabet_size=16, hidden_dim=12
    )
    coder = ScorerConditionalEntropyCoder(model)
    N = 30
    side = torch.randn(N, 6)
    symbols = [int(s) for s in torch.randint(0, 16, (N,)).tolist()]
    payload = coder.encode(symbols, side)
    section = serialize_jscc_section(
        payload=payload,
        side_dim=6,
        alphabet_size=16,
        n_symbols=N,
    )
    parsed = parse_jscc_section(section)
    decoded = coder.decode(parsed.payload, side, n_symbols=N)
    assert decoded == symbols
    assert parsed.manifest.payload_length == len(payload)
    assert parsed.manifest.total_section_bytes == len(section)


def test_manifest_typed_attributes_present():
    section = serialize_jscc_section(
        payload=b"\xaa\xbb", side_dim=3, alphabet_size=8, n_symbols=2
    )
    parsed = parse_jscc_section(section)
    m = parsed.manifest
    # All typed fields populated
    assert m.payload_offset > 0
    assert m.payload_length == 2
    assert m.total_section_bytes == m.payload_offset + m.payload_length
    assert m.custody_metadata_offset > 0
    assert m.custody_metadata_length > 0
    assert isinstance(m, JSCCSectionManifest)
    assert isinstance(parsed, JSCCArchiveSection)
