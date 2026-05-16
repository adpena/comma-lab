# SPDX-License-Identifier: MIT
"""Smoke tests for the Tishby IB-pure research-only package closure."""

from __future__ import annotations

from tac.substrates.tishby_ib_pure import (
    RESEARCH_ONLY,
    TIBP1_MAGIC,
    TishbyIBPureCodec,
    TishbyIBPureScoreAwareLoss,
    inflate_one_video,
    pack_archive,
    parse_archive,
)


def test_package_import_is_research_only() -> None:
    codec = TishbyIBPureCodec()
    summary = codec.encode_summary()
    assert RESEARCH_ONLY is True
    assert summary["score_claim"] is False
    assert summary["research_only"] is True


def test_tibp1_archive_roundtrip_and_inflate_consumes_bytes(tmp_path) -> None:
    archive = pack_archive({"decoder_blob": b"decoder", "latent_t_blob": b"latent"})
    assert archive.startswith(TIBP1_MAGIC)
    parsed = parse_archive(archive)
    assert parsed.sections["decoder_blob"] == b"decoder"
    assert parsed.sections["latent_t_blob"] == b"latent"
    out = inflate_one_video(archive, tmp_path / "inflate_proof.txt")
    assert "score_claim=False" in out.read_text(encoding="utf-8")


def test_loss_facade_is_decomposed_and_non_claiming() -> None:
    loss = TishbyIBPureScoreAwareLoss()
    output = loss(reconstruction_term=2.0, kl_term=3.0, rate_term=4.0)
    assert output.total == 2.0 + 0.01 * 3.0 + 4.0
    assert output.score_claim is False
