"""Tests for PR103 AC runtime closure inside PR106 packed envelopes."""

from __future__ import annotations

from dataclasses import replace

import brotli
import numpy as np
import pytest
import torch

from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA
from tac.pr103_arithmetic_codec import (
    AC_TENSOR_INDICES,
    EncodedAcDecoderBlob,
    encode_decoder_ac,
)
from tac.pr103_pr106_runtime_closure import (
    PR106_PACKED_META,
    Pr103Pr106RuntimeClosure,
    Pr103Pr106RuntimeClosureError,
    build_runtime_closure_from_layout,
    decode_pr106_packed_payload,
    derive_runtime_closure_from_pr106_source,
    parse_pr103_repacked_pr106_payload,
    split_pr106_packed_payload,
)


def _synthetic_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(*shape, generator=g) * 0.1
        for name, shape in FIXED_STATE_SCHEMA
    }


def _zigzag_encode_i8(arr_i8: np.ndarray) -> np.ndarray:
    arr = arr_i8.astype(np.int32)
    return np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)


def _encode_pr106_packed_decoder(sd: dict[str, torch.Tensor]) -> bytes:
    packed_schema = sorted(FIXED_STATE_SCHEMA, key=lambda item: -int(np.prod(item[1])))
    payloads: list[bytes] = []
    scales: list[float] = []
    for name, _shape in packed_schema:
        t = sd[name].detach().cpu().float()
        m = t.abs().max().item()
        scale = m / 127 if m > 0 else 1.0
        q = (t / scale).round().clamp(-127, 127).to(torch.int8).numpy().flatten()
        payloads.append(_zigzag_encode_i8(q).tobytes())
        scales.append(scale)
    raw = b"".join(payloads) + np.array(scales, dtype=np.float32).tobytes()
    return brotli.compress(raw, quality=11)


def _encode_pr106_fixed_latents(seed: int = 0) -> tuple[bytes, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(
        int(PR106_PACKED_META["n_pairs"]),
        int(PR106_PACKED_META["latent_dim"]),
        generator=g,
    )
    t = latents.detach().cpu().float()
    mins = t.min(dim=0).values
    maxs = t.max(dim=0).values
    scales = ((maxs - mins) / 254.0).clamp(min=1e-10)
    q = ((t - mins.unsqueeze(0)) / scales.unsqueeze(0)).round()
    q = q.clamp(0, 254).to(torch.uint8).numpy()
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0]
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    delta_zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    lo = (delta_zz & 0xFF).astype(np.uint8).tobytes()
    hi = (delta_zz >> 8).astype(np.uint8).tobytes()
    raw = (
        lo
        + mins.to(torch.float16).numpy().tobytes()
        + scales.to(torch.float16).numpy().tobytes()
        + hi
    )
    return brotli.compress(raw, quality=11), latents


def _packed_payload(decoder: bytes, latents: bytes) -> bytes:
    assert len(decoder) < (1 << 24)
    return b"\xff" + len(decoder).to_bytes(3, "little") + decoder + latents


def _pr103_layout(
    sd: dict[str, torch.Tensor],
    *,
    ac_auto_fallback: bool = False,
) -> EncodedAcDecoderBlob:
    layout = encode_decoder_ac(
        sd,
        adaptive_lgwin=False,
        return_layout=True,
        ac_auto_fallback=ac_auto_fallback,
    )
    assert isinstance(layout, EncodedAcDecoderBlob)
    return layout


def test_runtime_closure_decodes_pr103_decoder_inside_pr106_envelope() -> None:
    sd = _synthetic_state_dict()
    latents_brotli, _latents = _encode_pr106_fixed_latents()
    layout = _pr103_layout(sd)
    candidate_payload = _packed_payload(layout.blob, latents_brotli)

    closure = build_runtime_closure_from_layout(
        layout,
        candidate_payload=candidate_payload,
        adaptive_lgwin=False,
        ac_auto_fallback=False,
    )
    decoded = parse_pr103_repacked_pr106_payload(candidate_payload, closure)

    assert decoded.meta == PR106_PACKED_META
    assert tuple(decoded.latents.shape) == (600, 28)
    assert set(decoded.state_dict) == {name for name, _shape in FIXED_STATE_SCHEMA}
    layout_b = _pr103_layout(decoded.state_dict)
    assert layout_b.blob == layout.blob


def test_runtime_closure_derives_from_pr106_source_and_matches_candidate() -> None:
    sd = _synthetic_state_dict(seed=11)
    latents_brotli, _latents = _encode_pr106_fixed_latents(seed=12)
    source_payload = _packed_payload(_encode_pr106_packed_decoder(sd), latents_brotli)
    source_decoded = decode_pr106_packed_payload(source_payload)
    candidate_layout = _pr103_layout(source_decoded.state_dict)
    candidate_payload = _packed_payload(candidate_layout.blob, latents_brotli)

    closure = derive_runtime_closure_from_pr106_source(
        source_payload=source_payload,
        candidate_payload=candidate_payload,
        adaptive_lgwin=False,
        ac_auto_fallback=False,
    )

    assert closure.section_lengths["ac_fallback"] == 0
    assert closure.ac_fallback_set == ()
    sections = split_pr106_packed_payload(candidate_payload)
    assert closure.decoder_section_bytes == len(sections.decoder)
    assert parse_pr103_repacked_pr106_payload(candidate_payload, closure).latents.shape == (
        600,
        28,
    )


def test_runtime_closure_fails_closed_on_bad_section_lengths() -> None:
    sd = _synthetic_state_dict()
    latents_brotli, _latents = _encode_pr106_fixed_latents()
    layout = _pr103_layout(sd)
    candidate_payload = _packed_payload(layout.blob, latents_brotli)
    closure = build_runtime_closure_from_layout(
        layout,
        candidate_payload=candidate_payload,
        adaptive_lgwin=False,
        ac_auto_fallback=False,
    )
    bad_lengths = dict(closure.section_lengths)
    bad_lengths["merged_ac"] += 1
    bad = replace(closure, section_lengths=bad_lengths)

    with pytest.raises(Pr103Pr106RuntimeClosureError, match="section_lengths sum"):
        parse_pr103_repacked_pr106_payload(candidate_payload, bad)


def test_runtime_closure_fails_closed_on_decoder_sha_mismatch() -> None:
    sd = _synthetic_state_dict()
    latents_brotli, _latents = _encode_pr106_fixed_latents()
    layout = _pr103_layout(sd)
    candidate_payload = _packed_payload(layout.blob, latents_brotli)
    closure = build_runtime_closure_from_layout(
        layout,
        candidate_payload=candidate_payload,
        adaptive_lgwin=False,
        ac_auto_fallback=False,
    )
    bad = replace(closure, decoder_section_sha256="0" * 64)

    with pytest.raises(Pr103Pr106RuntimeClosureError, match="decoder SHA-256"):
        parse_pr103_repacked_pr106_payload(candidate_payload, bad)


def test_runtime_closure_fails_closed_on_invalid_fallback_index() -> None:
    with pytest.raises(Pr103Pr106RuntimeClosureError, match="non-PR103 AC indices"):
        Pr103Pr106RuntimeClosure(
            section_lengths=dict.fromkeys(
                ("br", "hists", "merged_ac", "hi_hist", "ac_fallback"), 0
            ),
            ac_fallback_set=(1,),
        )


def test_runtime_closure_requires_fallback_set_when_fallback_bytes_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.pr103_arithmetic_codec as pr103

    sd = _synthetic_state_dict()
    original_pack_ac_stream = pr103.pack_ac_stream

    def inflated_pack_ac_stream(symbols: np.ndarray, histogram: np.ndarray) -> bytes:
        return original_pack_ac_stream(symbols, histogram) + b"regression" * 2048

    monkeypatch.setattr(pr103, "pack_ac_stream", inflated_pack_ac_stream)
    layout = _pr103_layout(sd, ac_auto_fallback=True)
    assert layout.ac_fallback_set
    latents_brotli, _latents = _encode_pr106_fixed_latents()
    candidate_payload = _packed_payload(layout.blob, latents_brotli)
    closure = build_runtime_closure_from_layout(
        layout,
        candidate_payload=candidate_payload,
        adaptive_lgwin=False,
        ac_auto_fallback=True,
    )
    assert parse_pr103_repacked_pr106_payload(candidate_payload, closure).latents.shape == (
        600,
        28,
    )

    bad = replace(closure, ac_fallback_set=())
    with pytest.raises(Pr103Pr106RuntimeClosureError, match="ac_fallback section"):
        parse_pr103_repacked_pr106_payload(candidate_payload, bad)


def test_runtime_closure_rejects_fallback_set_without_fallback_bytes() -> None:
    sd = _synthetic_state_dict()
    latents_brotli, _latents = _encode_pr106_fixed_latents()
    layout = _pr103_layout(sd)
    candidate_payload = _packed_payload(layout.blob, latents_brotli)
    closure = build_runtime_closure_from_layout(
        layout,
        candidate_payload=candidate_payload,
        adaptive_lgwin=False,
        ac_auto_fallback=False,
    )
    bad = replace(closure, ac_fallback_set=(AC_TENSOR_INDICES[0],))

    with pytest.raises(Pr103Pr106RuntimeClosureError, match="ac_fallback_set"):
        parse_pr103_repacked_pr106_payload(candidate_payload, bad)
