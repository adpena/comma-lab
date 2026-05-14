# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

import pytest

from tac.codec.jscc import (
    CONFORMANCE_VECTORS,
    LEGACY_JSCC_HUFFMAN_MAGIC,
    JSCCCodingContext,
    JSCCSection,
    ScorerConditionalHuffmanCoder,
    ScorerConditionalSignal,
    allocate_scorer_conditional_bytes,
)


def test_jscc_allocation_is_deterministic_and_sums_exactly() -> None:
    sections = [
        JSCCSection("pose", raw_bytes=10, scorer_weight=4.0),
        JSCCSection("mask", raw_bytes=30, scorer_weight=1.0),
        JSCCSection("renderer", raw_bytes=20, scorer_weight=2.0),
    ]

    allocation = allocate_scorer_conditional_bytes(sections, total_bytes=17)

    assert sum(allocation.values()) == 17
    assert allocation == {"pose": 6, "mask": 5, "renderer": 6}


def test_jscc_huffman_roundtrips_and_keeps_non_authority_metadata() -> None:
    signal = ScorerConditionalSignal("seg_margin", (0.1, 0.5), kind="sensitivity")
    context = JSCCCodingContext(section_name="renderer.bin", signals=(signal,))
    coder = ScorerConditionalHuffmanCoder()

    encoded = coder.encode(b"abracadabra abracadabra", context=context)
    decoded = coder.decode(encoded.data)

    assert decoded == b"abracadabra abracadabra"
    assert encoded.score_claim is False
    assert encoded.promotion_eligible is False
    assert encoded.ready_for_exact_eval_dispatch is False
    assert encoded.proxy_only is True
    metadata = context.manifest_metadata()
    assert metadata["legacy_huffman_magic"] == "JSCC"
    assert metadata["score_claim"] is False
    assert metadata["promotion_eligible"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    assert metadata["proxy"] is True
    assert metadata["proxy_only"] is True


def test_jscc_conformance_vector_is_pinned() -> None:
    vector = CONFORMANCE_VECTORS[0]
    context = JSCCCodingContext(section_name=str(vector["section_name"]))
    coder = ScorerConditionalHuffmanCoder(prior_strength=float(vector["prior_strength"]))
    encoded = coder.encode(bytes.fromhex(str(vector["payload_hex"])), context=context)

    assert encoded.data.hex() == vector["packet_hex"]
    assert hashlib.sha256(encoded.data).hexdigest() == vector["packet_sha256"]
    assert coder.decode(bytes.fromhex(str(vector["packet_hex"]))) == bytes.fromhex(
        str(vector["payload_hex"])
    )


def test_jscc_rejects_score_claims_bad_priors_and_corrupt_packets() -> None:
    assert LEGACY_JSCC_HUFFMAN_MAGIC == b"JSCC"
    with pytest.raises(ValueError, match="score_claim"):
        ScorerConditionalSignal("bad", (1.0,), score_claim=True)
    with pytest.raises(ValueError, match="score_claim"):
        JSCCCodingContext("bad", score_claim=True)
    with pytest.raises(ValueError, match="ready_for_exact_eval_dispatch"):
        JSCCCodingContext("bad", ready_for_exact_eval_dispatch=True)

    coder = ScorerConditionalHuffmanCoder(prior_strength=1.0)
    context = JSCCCodingContext("payload")
    with pytest.raises(ValueError, match="256"):
        coder.encode(b"abc", context=context, symbol_prior=[1.0, 2.0])

    encoded = bytearray(ScorerConditionalHuffmanCoder().encode(b"abc", context=context).data)
    encoded[0:4] = b"BAD!"
    with pytest.raises(ValueError, match="magic"):
        ScorerConditionalHuffmanCoder.decode(bytes(encoded))
