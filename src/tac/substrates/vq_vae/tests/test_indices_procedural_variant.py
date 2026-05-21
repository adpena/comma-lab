# SPDX-License-Identifier: MIT
"""Tests for the VQ-VAE procedural-index residual scaffold."""

from __future__ import annotations

import struct

import pytest
import torch

from tac.substrates.vq_vae import (
    PROCEDURAL_INDICES_RESIDUAL_CONTEXT,
    ProceduralIndicesVariantError,
    analyze_procedural_indices_blob,
    compose_with_procedural_indices,
    decode_procedural_indices_blob,
    derive_procedural_indices_predictor,
    encode_procedural_indices_blob,
    parse_archive,
)
from tac.substrates.vq_vae.archive import (
    VQV1_HEADER_FMT,
    VQV1_HEADER_SIZE,
    VQV1_PROCEDURAL_INDICES_SENTINEL,
)
from tac.substrates.vq_vae.tests.test_procedural_variant import (
    _CANONICAL_SEED_32B,
    _make_synthetic_vqv1_archive_bytes,
)


def _indices() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[0, 1, 2], [3, 4, 5]],
                [[6, 7, 8], [9, 10, 11]],
            ],
            [
                [[12, 13, 14], [15, 0, 1]],
                [[2, 3, 4], [5, 6, 7]],
            ],
        ],
        dtype=torch.int64,
    )


def test_residual_context_routes_to_residual_equation() -> None:
    """The VQPI scaffold is explicitly residual-correction, not equation #26."""

    assert "_residual_correction_" in PROCEDURAL_INDICES_RESIDUAL_CONTEXT


def test_derive_procedural_indices_predictor_is_deterministic() -> None:
    """Same seed and shape produce identical bounded index predictions."""

    a = derive_procedural_indices_predictor(
        _CANONICAL_SEED_32B,
        shape=(2, 2, 2, 3),
        codebook_size=16,
    )
    b = derive_procedural_indices_predictor(
        _CANONICAL_SEED_32B,
        shape=(2, 2, 2, 3),
        codebook_size=16,
    )
    assert torch.equal(a, b)
    assert int(a.min()) >= 0
    assert int(a.max()) < 16


def test_derive_procedural_indices_predictor_seed_mutation_changes_prediction() -> None:
    """Changing the seed changes the predictor surface."""

    seed_b = bytearray(_CANONICAL_SEED_32B)
    seed_b[0] ^= 0xFF
    a = derive_procedural_indices_predictor(
        _CANONICAL_SEED_32B,
        shape=(2, 2, 2, 3),
        codebook_size=16,
    )
    b = derive_procedural_indices_predictor(
        bytes(seed_b),
        shape=(2, 2, 2, 3),
        codebook_size=16,
    )
    assert not torch.equal(a, b)


def test_encode_decode_procedural_indices_roundtrip_exact() -> None:
    """Seed plus residual reconstructs the original indices exactly."""

    source = _indices()
    blob = encode_procedural_indices_blob(
        source,
        codebook_size=16,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    decoded = decode_procedural_indices_blob(
        blob,
        shape=tuple(source.shape),
        codebook_size=16,
    )
    assert torch.equal(decoded, source)


def test_analyze_procedural_indices_blob_uses_residual_equation() -> None:
    """Byte accounting is rate-only and non-promotional."""

    result = analyze_procedural_indices_blob(
        _indices(),
        codebook_size=16,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    assert result["equation_id"] == "procedural_predictor_plus_residual_correction_savings_v1"
    assert result["score_claim"] is False
    accounting = result["predicted_rate_accounting"]
    assert isinstance(accounting, dict)
    assert accounting["context"] == PROCEDURAL_INDICES_RESIDUAL_CONTEXT


def test_compose_with_procedural_indices_parses_and_preserves_indices() -> None:
    """A VQPI archive parses through the normal VQV1 parser."""

    original = _make_synthetic_vqv1_archive_bytes()
    original_arc = parse_archive(original)
    composed = compose_with_procedural_indices(
        original_archive_bytes=original,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    parsed = parse_archive(composed.archive_bytes)
    assert torch.equal(parsed.indices, original_arc.indices)
    assert parsed.meta == original_arc.meta
    assert composed.indices_blob.startswith(VQV1_PROCEDURAL_INDICES_SENTINEL)
    assert composed.predicted_rate_accounting["score_claim"] is False


def test_compose_with_procedural_indices_records_honest_byte_delta() -> None:
    """Small synthetic residuals may grow; the scaffold records that honestly."""

    original = _make_synthetic_vqv1_archive_bytes()
    composed = compose_with_procedural_indices(
        original_archive_bytes=original,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    accounting = composed.predicted_rate_accounting
    assert accounting["delta_bytes_replacement_minus_original"] == (
        composed.replacement_total_bytes - composed.original_indices_bytes
    )
    assert accounting["verdict"] in {"RATE_WIN", "RATE_REGRESSION", "RATE_NEUTRAL"}


def test_mutating_seed_inside_envelope_changes_decoded_indices() -> None:
    """The seed is operationally consumed; mutating it changes decoded indices."""

    original = _make_synthetic_vqv1_archive_bytes()
    composed = compose_with_procedural_indices(
        original_archive_bytes=original,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    header = struct.unpack(VQV1_HEADER_FMT, composed.archive_bytes[:VQV1_HEADER_SIZE])
    decoder_len = header[7]
    seed_offset = VQV1_HEADER_SIZE + decoder_len + 12
    mutated = bytearray(composed.archive_bytes)
    mutated[seed_offset] ^= 0xFF
    parsed_original = parse_archive(composed.archive_bytes)
    parsed_mutated = parse_archive(bytes(mutated))
    assert not torch.equal(parsed_original.indices, parsed_mutated.indices)


def test_decode_procedural_indices_rejects_bad_sentinel() -> None:
    """Malformed raw bytes do not silently decode as procedural indices."""

    blob = b"BAD!" + b"\x01\x00\x20\x00\x00\x00\x00\x00"
    with pytest.raises(ProceduralIndicesVariantError, match="bad procedural indices sentinel"):
        decode_procedural_indices_blob(blob, shape=(1, 2, 1, 1), codebook_size=16)


def test_decode_procedural_indices_rejects_wrong_shape_residual() -> None:
    """Residual payload length must match the parser-provided VQV1 shape."""

    source = _indices()
    blob = encode_procedural_indices_blob(
        source,
        codebook_size=16,
        seed_bytes=_CANONICAL_SEED_32B,
    )
    with pytest.raises(ProceduralIndicesVariantError, match="residual raw length"):
        decode_procedural_indices_blob(blob, shape=(1, 2, 2, 3), codebook_size=16)


def test_base_raw_indices_archive_still_parses() -> None:
    """The VQPI parser extension does not regress canonical raw indices."""

    original = _make_synthetic_vqv1_archive_bytes()
    parsed = parse_archive(original)
    assert parsed.indices.shape == (3, 2, 2, 3)


def test_trainer_declares_default_off_procedural_indices_flags(tmp_path) -> None:
    """Trainer exposes VQPI as opt-in research-only archive build knob."""

    from experiments.train_substrate_vq_vae import (
        _build_parser,
        _derive_procedural_indices_seed,
    )

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--epochs",
            "1",
            "--smoke",
            "--device",
            "cpu",
            "--enable-procedural-indices-residual",
            "--procedural-indices-seed-bytes",
            "24",
            "--procedural-indices-generator-kind",
            "xorshift",
        ]
    )
    assert args.enable_procedural_indices_residual is True
    seed_a = _derive_procedural_indices_seed(args)
    seed_b = _derive_procedural_indices_seed(args)
    assert seed_a == seed_b
    assert len(seed_a) == 24
