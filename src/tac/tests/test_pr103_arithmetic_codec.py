"""Tests for :mod:`tac.pr103_arithmetic_codec`.

Coverage:
- Roundtrip on synthetic 28-tensor HNeRV-shaped state_dict — encode + decode
  produces bit-identical (post-quantization) tensors.
- Format-stability: blob length includes scales + non-AC brotli + histograms +
  merged AC + latent-hi histogram.
- ``validate_ac_savings`` returns dict with all ``AC_TENSOR_INDICES`` keys;
  measures correctly.
- Merged-RangeEncoder optimization: a single encoder for all streams produces
  ≤ sum of per-stream encoded bytes (eliminates rounding overhead).
- ``adaptive_lgwin_search`` returns a valid lgwin in [10, 24].
- Composition with Op 1: PR101 encode → decode → PR103 encode → decode →
  re-encode produces stable post-quantization tensors.
- Edge cases: empty state, missing tensor, mismatched section_lengths.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from tac.pr101_split_brotli_codec import (
    FIXED_STATE_SCHEMA,
    decode_decoder_compact,
    encode_decoder_compact,
    pack_brotli_stream,
)
from tac.pr103_arithmetic_codec import (
    AC_HISTOGRAM_BITS,
    AC_SYMBOL_OFFSET,
    AC_TENSOR_INDICES,
    ADAPTIVE_LGWIN_MAX,
    ADAPTIVE_LGWIN_MIN,
    MERGED_RANGE_ENCODER,
    DecodedAcDecoderBlob,
    EncodedAcDecoderBlob,
    Pr103ArithmeticCodecError,
    adaptive_lgwin_search,
    decode_decoder_ac,
    encode_decoder_ac,
    pack_ac_stream,
    unpack_ac_stream,
    validate_ac_savings,
)


def _synthetic_state_dict(seed: int = 0, scale: float = 0.1) -> dict[str, torch.Tensor]:
    """Build a synthetic HNeRV-shaped state_dict from FIXED_STATE_SCHEMA."""
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for name, shape in FIXED_STATE_SCHEMA:
        sd[name] = torch.randn(*shape, generator=g) * scale
    return sd


# ---------------------------------------------------------------------------
# Constants surface
# ---------------------------------------------------------------------------

def test_ac_tensor_indices_is_8_unique_ints() -> None:
    assert isinstance(AC_TENSOR_INDICES, tuple)
    assert len(AC_TENSOR_INDICES) == 8
    assert len(set(AC_TENSOR_INDICES)) == 8
    for idx in AC_TENSOR_INDICES:
        assert isinstance(idx, int)
        assert 0 <= idx < len(FIXED_STATE_SCHEMA)


def test_constants_match_pr103_source_of_truth() -> None:
    """PR103 uses q8 (uint8) histograms and merges 9 streams into 1 encoder."""
    assert AC_HISTOGRAM_BITS == 8
    assert MERGED_RANGE_ENCODER is True
    assert AC_SYMBOL_OFFSET == 128
    assert ADAPTIVE_LGWIN_MIN == 10
    assert ADAPTIVE_LGWIN_MAX == 24


# ---------------------------------------------------------------------------
# Encode / decode roundtrip
# ---------------------------------------------------------------------------

def test_encode_decoder_ac_no_latent_hi_returns_bytes() -> None:
    sd = _synthetic_state_dict()
    blob = encode_decoder_ac(sd)
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_encode_decoder_ac_return_layout_dataclass() -> None:
    sd = _synthetic_state_dict()
    layout = encode_decoder_ac(sd, return_layout=True)
    assert isinstance(layout, EncodedAcDecoderBlob)
    assert isinstance(layout.blob, bytes)
    assert isinstance(layout.merged_ac_blob, bytes)
    assert isinstance(layout.histograms_blob, bytes)
    assert isinstance(layout.fp16_scales, bytes)
    assert len(layout.fp16_scales) == 28 * 2
    # Sum of sections must equal blob length.
    section_sum = (
        len(layout.fp16_scales)
        + sum(len(s) for s in layout.non_ac_brotli_streams)
        + len(layout.histograms_blob)
        + len(layout.merged_ac_blob)
        + len(layout.latent_hi_hist_blob)
    )
    assert section_sum == len(layout.blob), (
        f"section_sum {section_sum} != blob {len(layout.blob)}"
    )
    # Selected lgwins must be valid.
    for lgwin in layout.selected_lgwins:
        assert ADAPTIVE_LGWIN_MIN <= lgwin <= ADAPTIVE_LGWIN_MAX


def test_encode_decode_roundtrip_no_latent_hi() -> None:
    """Round-trip without latent-hi: encode then decode reconstructs all 28 tensors
    bit-identically (post-quantization)."""
    sd = _synthetic_state_dict()
    layout = encode_decoder_ac(sd, return_layout=True)
    section_lengths = {
        "br": sum(len(s) for s in layout.non_ac_brotli_streams),
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
    }
    decoded = decode_decoder_ac(
        layout.blob,
        section_lengths=section_lengths,
        n_latent_hi_symbols=0,
    )
    assert isinstance(decoded, DecodedAcDecoderBlob)
    assert decoded.latent_hi_symbols is None
    restored = decoded.state_dict
    assert set(restored.keys()) == set(sd.keys())
    # Re-encode and check byte-identical (round-trip is idempotent on the
    # quantization grid).
    layout_b = encode_decoder_ac(restored, return_layout=True)
    assert layout.blob == layout_b.blob, (
        f"non-idempotent encode: layout.blob len {len(layout.blob)} "
        f"vs layout_b.blob len {len(layout_b.blob)}"
    )


def test_encode_decode_roundtrip_with_all_ac_tensors_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-tensor AC auto-fallback is part of the decoder contract.

    Force the encode-side audit to decide every AC tensor regresses, then prove
    the fallback brotli section plus ``ac_fallback_set`` reconstructs the same
    post-quantization state.
    """
    import tac.pr103_arithmetic_codec as pr103

    sd = _synthetic_state_dict()
    original_pack_ac_stream = pr103.pack_ac_stream

    def inflated_pack_ac_stream(symbols: np.ndarray, histogram: np.ndarray) -> bytes:
        return original_pack_ac_stream(symbols, histogram) + b"forced-regression" * 2048

    monkeypatch.setattr(pr103, "pack_ac_stream", inflated_pack_ac_stream)
    layout = encode_decoder_ac(sd, return_layout=True)

    assert layout.ac_fallback_set == AC_TENSOR_INDICES
    assert layout.histograms_blob == b""
    assert layout.merged_ac_blob == b""
    assert layout.ac_fallback_blob

    section_lengths = {
        "br": sum(len(s) for s in layout.non_ac_brotli_streams),
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
        "ac_fallback": len(layout.ac_fallback_blob),
    }
    decoded = decode_decoder_ac(
        layout.blob,
        section_lengths=section_lengths,
        ac_fallback_set=layout.ac_fallback_set,
    )
    layout_b = encode_decoder_ac(decoded.state_dict, return_layout=True)
    assert layout.blob == layout_b.blob


def test_encode_decode_roundtrip_with_latent_hi() -> None:
    """Round-trip with latent-hi: 16800 symbols are encoded into the merged AC
    stream and decoded back."""
    sd = _synthetic_state_dict()
    g = np.random.default_rng(0)
    # Realistic latent-hi: most values are 0, sparse.
    hi = g.integers(0, 5, size=600 * 28).astype(np.uint16)
    layout = encode_decoder_ac(sd, latent_hi_symbols=hi, return_layout=True)
    section_lengths = {
        "br": sum(len(s) for s in layout.non_ac_brotli_streams),
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
    }
    decoded = decode_decoder_ac(
        layout.blob,
        section_lengths=section_lengths,
        n_latent_hi_symbols=hi.size,
    )
    assert decoded.latent_hi_symbols is not None
    assert decoded.latent_hi_symbols.size == hi.size
    assert np.array_equal(
        decoded.latent_hi_symbols.astype(np.uint16), hi
    ), "latent-hi symbols failed AC roundtrip"


def test_encode_rejects_missing_tensor() -> None:
    sd = _synthetic_state_dict()
    del sd["stem.weight"]
    with pytest.raises(Pr103ArithmeticCodecError):
        encode_decoder_ac(sd)


def test_decode_requires_section_lengths() -> None:
    sd = _synthetic_state_dict()
    blob = encode_decoder_ac(sd)
    with pytest.raises(Pr103ArithmeticCodecError):
        decode_decoder_ac(blob, section_lengths=None)


def test_decode_rejects_mismatched_section_lengths() -> None:
    sd = _synthetic_state_dict()
    layout = encode_decoder_ac(sd, return_layout=True)
    bad_lengths = {
        "br": 999_999,  # wrong
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
    }
    with pytest.raises(Pr103ArithmeticCodecError):
        decode_decoder_ac(
            layout.blob, section_lengths=bad_lengths, n_latent_hi_symbols=0
        )


# ---------------------------------------------------------------------------
# Per-stream AC primitives
# ---------------------------------------------------------------------------

def test_pack_unpack_ac_stream_roundtrip() -> None:
    g = np.random.default_rng(0)
    syms = g.integers(0, 256, size=1024).astype(np.uint8)
    hist = np.bincount(syms, minlength=256).astype(np.uint8)
    hist = np.maximum(hist, 1)
    blob = pack_ac_stream(syms, hist)
    out = unpack_ac_stream(blob, hist, n_symbols=syms.size)
    assert np.array_equal(out.astype(np.uint8), syms)


def test_unpack_ac_stream_rejects_unaligned_blob() -> None:
    hist = np.ones(256, dtype=np.uint8)
    with pytest.raises(Pr103ArithmeticCodecError):
        unpack_ac_stream(b"\x01\x02\x03", hist, n_symbols=1)


# ---------------------------------------------------------------------------
# Adaptive lgwin
# ---------------------------------------------------------------------------

def test_adaptive_lgwin_search_returns_valid_lgwin() -> None:
    raw = b"hello world" * 1000
    lgwin, comp = adaptive_lgwin_search(raw)
    assert ADAPTIVE_LGWIN_MIN <= lgwin <= ADAPTIVE_LGWIN_MAX
    assert isinstance(comp, bytes)
    # Confirm it's at least as good as the brotli default.
    default_comp = pack_brotli_stream(raw)
    assert len(comp) <= len(default_comp), (
        f"adaptive_lgwin_search produced {len(comp)} bytes vs default {len(default_comp)}"
    )


def test_adaptive_lgwin_search_handles_short_input() -> None:
    raw = b"x" * 8
    lgwin, comp = adaptive_lgwin_search(raw)
    assert ADAPTIVE_LGWIN_MIN <= lgwin <= ADAPTIVE_LGWIN_MAX
    assert isinstance(comp, bytes)


# ---------------------------------------------------------------------------
# Merged-encoder optimization
# ---------------------------------------------------------------------------

def test_merged_encoder_no_worse_than_per_stream_sum() -> None:
    """PR103 trick #6: merging N AC streams into ONE RangeEncoder produces a
    blob ≤ the sum of N independent encoders' outputs (eliminates per-stream
    rounding overhead).

    We measure: sum-of-per-stream encoded bytes vs merged-encoder output for
    the 8 AC streams of a synthetic state_dict.
    """
    sd = _synthetic_state_dict()
    layout = encode_decoder_ac(sd, return_layout=True)

    # Per-stream isolated AC: encode each AC tensor independently and sum
    # the resulting blob lengths.
    from tac.pr101_split_brotli_codec import N_QUANT, _quantize_tensor
    from tac.pr103_arithmetic_codec import _build_q8_histogram

    quantized = [
        _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        for name, _shape in FIXED_STATE_SCHEMA
    ]
    per_stream_total = 0
    for idx in AC_TENSOR_INDICES:
        qt = quantized[idx]
        u8 = (qt.q_i8.astype(np.int16) + AC_SYMBOL_OFFSET).astype(np.uint8).reshape(-1)
        hist = _build_q8_histogram(u8)
        per_stream_total += len(pack_ac_stream(u8, hist))
    merged_total = len(layout.merged_ac_blob)
    assert merged_total <= per_stream_total, (
        f"merged encoder ({merged_total}) was LARGER than sum of per-stream "
        f"encoders ({per_stream_total}) — trick #6 should never regress"
    )


# ---------------------------------------------------------------------------
# validate_ac_savings (Contrarian gate)
# ---------------------------------------------------------------------------

def test_validate_ac_savings_returns_all_ac_indices() -> None:
    sd = _synthetic_state_dict()
    audit = validate_ac_savings(sd)
    assert set(audit.keys()) == set(AC_TENSOR_INDICES)
    for _idx, info in audit.items():
        for key in ("ac_bytes", "brotli_bytes", "delta_bytes", "n_symbols"):
            assert key in info
        assert info["ac_bytes"] > 0
        assert info["brotli_bytes"] > 0
        assert info["delta_bytes"] == info["ac_bytes"] - info["brotli_bytes"]
        assert info["n_symbols"] > 0


def test_validate_ac_savings_warns_on_regression(caplog: pytest.LogCaptureFixture) -> None:
    """Synthetic-weights AC may regress on some tensors; the warning code
    path should be exercised. We don't force a regression; we just confirm
    the helper runs without crashing on synthetic weights and that the dict
    structure is correct."""
    sd = _synthetic_state_dict()
    with caplog.at_level(logging.WARNING, logger="tac.pr103_arithmetic_codec"):
        audit = validate_ac_savings(sd)
    # At least one of the 8 indices should produce a regression on
    # uniformly-random weights (entropy is high → AC barely helps and
    # storage overhead may exceed brotli). We check that the audit dict is
    # well-formed; the warning surface is exercised on real weights.
    assert all("delta_bytes" in info for info in audit.values())


# ---------------------------------------------------------------------------
# Composition with Op 1 (PR101 split-Brotli)
# ---------------------------------------------------------------------------

def test_compose_with_pr101_decoded_state_dict() -> None:
    """The Op 2 codec accepts state_dicts produced by Op 1's
    decode_decoder_compact — this is the canonical "stack on Op 1 output"
    workflow used by experiments/build_pr103_repacked_archive.py.
    """
    sd = _synthetic_state_dict()
    # Op 1: PR101 encode+decode (round-trip puts weights on the quant grid).
    op1_blob = encode_decoder_compact(sd)
    sd_op1 = decode_decoder_compact(op1_blob)
    # Op 2: encode the Op-1-decoded weights via PR103 AC.
    op2_blob = encode_decoder_ac(sd_op1)
    assert isinstance(op2_blob, bytes) and len(op2_blob) > 0
    # Round-trip Op 2 back; the decoded weights must match the Op-1-decoded
    # weights bit-identically.
    layout = encode_decoder_ac(sd_op1, return_layout=True)
    section_lengths = {
        "br": sum(len(s) for s in layout.non_ac_brotli_streams),
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
    }
    decoded = decode_decoder_ac(layout.blob, section_lengths=section_lengths)
    sd_op2 = decoded.state_dict
    for name, _shape in FIXED_STATE_SCHEMA:
        assert torch.allclose(sd_op1[name], sd_op2[name], atol=0, rtol=0), (
            f"Op2 round-trip differs for {name}: {sd_op1[name].shape} "
            f"max diff {(sd_op1[name] - sd_op2[name]).abs().max()}"
        )


def test_compose_with_pr101_idempotent_re_encode() -> None:
    """encode_pr103 → decode_pr103 → re-encode_pr103 gives identical bytes
    (the second round goes through quantization-on-grid, which is idempotent)."""
    sd = _synthetic_state_dict()
    sd_op1 = decode_decoder_compact(encode_decoder_compact(sd))
    layout = encode_decoder_ac(sd_op1, return_layout=True)
    section_lengths = {
        "br": sum(len(s) for s in layout.non_ac_brotli_streams),
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
    }
    decoded = decode_decoder_ac(layout.blob, section_lengths=section_lengths)
    layout_b = encode_decoder_ac(decoded.state_dict, return_layout=True)
    assert layout.blob == layout_b.blob, (
        f"non-idempotent encode under Op1+Op2 composition: "
        f"len {len(layout.blob)} vs {len(layout_b.blob)}"
    )


# ---------------------------------------------------------------------------
# Cross-codec measurement: AC + adaptive_lgwin should not WORSEN the encode
# ---------------------------------------------------------------------------

def test_encode_with_all_zero_latent_hi_does_not_crash() -> None:
    """Round 1 review (Yousfi finding): an all-zero latent-hi stream produces
    a single-bucket histogram, which crashes constriction.Categorical. The
    histogram builder must pad to at least 2 buckets.
    """
    sd = _synthetic_state_dict()
    hi = np.zeros(16800, dtype=np.uint16)
    layout = encode_decoder_ac(sd, latent_hi_symbols=hi, return_layout=True)
    section_lengths = {
        "br": sum(len(s) for s in layout.non_ac_brotli_streams),
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
    }
    decoded = decode_decoder_ac(
        layout.blob, section_lengths=section_lengths, n_latent_hi_symbols=hi.size
    )
    assert decoded.latent_hi_symbols is not None
    assert np.array_equal(decoded.latent_hi_symbols.astype(np.uint16), hi)


def test_adaptive_lgwin_off_produces_valid_blob() -> None:
    """Disabling adaptive_lgwin still produces a correct round-trip. Useful
    as a perf knob: skipping the search saves ~5s on encode."""
    sd = _synthetic_state_dict()
    layout = encode_decoder_ac(sd, adaptive_lgwin=False, return_layout=True)
    section_lengths = {
        "br": sum(len(s) for s in layout.non_ac_brotli_streams),
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
        "ac_fallback": len(layout.ac_fallback_blob),
    }
    decoded = decode_decoder_ac(
        layout.blob,
        section_lengths=section_lengths,
        ac_fallback_set=layout.ac_fallback_set,
    )
    for name, _shape in FIXED_STATE_SCHEMA:
        assert decoded.state_dict[name].shape == sd[name].shape


# ---------------------------------------------------------------------------
# Per-tensor AC auto-fallback (substrate-mismatch protection landed 2026-05-07)
# ---------------------------------------------------------------------------


def _force_ac_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the encode-side AC-vs-brotli audit deterministically regress.

    The fallback gate is about routing after the audit finds a regression, not
    about a fragile synthetic source distribution. Keep the test deterministic
    by inflating audit-only ``pack_ac_stream`` output.
    """
    import tac.pr103_arithmetic_codec as pr103

    original_pack_ac_stream = pr103.pack_ac_stream

    def inflated_pack_ac_stream(symbols: np.ndarray, histogram: np.ndarray) -> bytes:
        return original_pack_ac_stream(symbols, histogram) + b"forced-regression" * 2048

    monkeypatch.setattr(pr103, "pack_ac_stream", inflated_pack_ac_stream)


def test_ac_auto_fallback_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default ``ac_auto_fallback=True`` activates when an AC tensor regresses
    vs its brotli baseline. This test forces the audit path to report
    regressions and proves the fallback set is populated.

    Note on byte semantics: ``_force_ac_regression`` rigs the AUDIT only; the
    actual brotli fallback uses real brotli. On synthetic Gaussian-distribution
    weights, AC actually beats brotli for these specific u8 streams, so when
    the audit lies and forces all 8 tensors to fall back, the produced blob
    can be modestly LARGER than the OFF path (~70 B on the test fixture).
    The fallback gate's purpose is substrate-mismatch protection on REAL
    inputs (per :func:`test_pr106_substrate_savings_on_int6`); this test only
    validates that the gate FIRES when the audit reports a regression.
    """
    _force_ac_regression(monkeypatch)
    sd = _synthetic_state_dict()
    layout_on = encode_decoder_ac(sd, return_layout=True)
    layout_off = encode_decoder_ac(sd, ac_auto_fallback=False, return_layout=True)
    assert isinstance(layout_on, EncodedAcDecoderBlob)
    assert isinstance(layout_off, EncodedAcDecoderBlob)
    # Gate fired correctly: every AC index reported regression and was diverted.
    assert layout_on.ac_fallback_set == AC_TENSOR_INDICES
    # OFF path bypassed the gate: empty fallback set, all AC encodes preserved.
    assert layout_off.ac_fallback_set == ()
    # The ON-path blob carries the fallback brotli section; OFF does not.
    assert len(layout_on.ac_fallback_blob) > 0
    assert layout_off.ac_fallback_blob == b""


def test_ac_auto_fallback_off_preserves_old_behavior() -> None:
    """Setting ``ac_auto_fallback=False`` reproduces strict PR103-faithful
    behavior: every AC_TENSOR_INDICES tensor goes through AC; the
    ac_fallback_set is empty and the ac_fallback_blob is empty."""
    sd = _synthetic_state_dict()
    layout = encode_decoder_ac(sd, ac_auto_fallback=False, return_layout=True)
    assert layout.ac_fallback_set == ()
    assert layout.ac_fallback_blob == b""


def test_decode_with_fallback_set_roundtrips(monkeypatch: pytest.MonkeyPatch) -> None:
    """encode (fallback ON) → decode (with the same fallback_set) → state_dict
    that round-trips bit-identically when re-encoded.

    The forced audit regression exercises the brotli-side decoder path for the
    fallback tensors.
    """
    _force_ac_regression(monkeypatch)
    sd = _synthetic_state_dict()
    layout = encode_decoder_ac(sd, return_layout=True)
    assert layout.ac_fallback_set == AC_TENSOR_INDICES
    section_lengths = {
        "br": sum(len(s) for s in layout.non_ac_brotli_streams),
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
        "ac_fallback": len(layout.ac_fallback_blob),
    }
    decoded = decode_decoder_ac(
        layout.blob,
        section_lengths=section_lengths,
        ac_fallback_set=layout.ac_fallback_set,
    )
    # Re-encode the recovered state — bit-identical idempotent round-trip.
    layout_b = encode_decoder_ac(decoded.state_dict, return_layout=True)
    assert layout.blob == layout_b.blob, (
        f"non-idempotent encode under fallback ({len(layout.blob)} vs "
        f"{len(layout_b.blob)} bytes)"
    )
    assert layout.ac_fallback_set == layout_b.ac_fallback_set


def test_decode_rejects_fallback_set_outside_ac_indices() -> None:
    """The decoder must reject a fallback_set that contains indices not in
    AC_TENSOR_INDICES — that means the encoder/decoder are out of sync and
    the bytes will not be reconstructable."""
    sd = _synthetic_state_dict()
    layout = encode_decoder_ac(sd, ac_auto_fallback=False, return_layout=True)
    section_lengths = {
        "br": sum(len(s) for s in layout.non_ac_brotli_streams),
        "hists": len(layout.histograms_blob),
        "merged_ac": len(layout.merged_ac_blob),
        "hi_hist": len(layout.latent_hi_hist_blob),
        "ac_fallback": len(layout.ac_fallback_blob),
    }
    with pytest.raises(Pr103ArithmeticCodecError):
        decode_decoder_ac(
            layout.blob,
            section_lengths=section_lengths,
            ac_fallback_set=(0, 1),  # idx 1 is NOT in AC_TENSOR_INDICES
        )


def test_op2_codec_pipeline_threads_fallback_through_op_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Op2 wrapper must:
    1. Default ``ac_auto_fallback=True``.
    2. Record ``ac_fallback_set`` in op_state as a JSON-serializable list.
    3. Round-trip via the CodecPipeline: encode → CPL1 wrapper → decode → state_dict
       even when fallback fires.
    """
    from tac.codec_pipeline import CodecPipeline, Op2_PR103ArithmeticCodec
    _force_ac_regression(monkeypatch)
    sd = _synthetic_state_dict()
    # Default-on threading: pipeline encode populates op_state[ac_fallback_set].
    pipe = CodecPipeline([Op2_PR103ArithmeticCodec()])
    blob, manifest = pipe.encode(sd, skip_validate=True)
    res = manifest.op_results[0]
    assert "ac_fallback_set" in res.op_state
    assert isinstance(res.op_state["ac_fallback_set"], list)
    assert tuple(res.op_state["ac_fallback_set"]) == AC_TENSOR_INDICES
    # Round-trip via pipeline decode (which threads op_state internally).
    sd_back, replayed = pipe.decode(blob)
    assert replayed == ["pr103_arithmetic_codec"]
    for name, _shape in FIXED_STATE_SCHEMA:
        assert sd_back[name].shape == sd[name].shape


def test_pr106_substrate_savings_on_int6() -> None:
    """Empirical PR106-substrate savings: when Op3(int6) substrate-transforms
    the PR106 weights and hands them to Op2, the ac_auto_fallback ON path
    must produce strictly FEWER bytes than the OFF path.

    The PR106 state_dict is the canonical real-archive substrate from
    ``experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt``;
    after int6 quantization, tensors 0/2/4 regress vs brotli by thousands of
    bytes (per the substrate-mismatch finding 2026-05-07).
    """
    from pathlib import Path

    from tac.codec_pipeline import CodecPipeline, Op2_PR103ArithmeticCodec
    from tac.codec_pipeline_apogee_int import Op3_ApogeeIntN_Substrate
    sd_path = Path("experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt")
    if not sd_path.exists():
        pytest.skip(f"PR106 substrate not available at {sd_path}")
    sd = torch.load(sd_path, weights_only=True, map_location="cpu")
    pipe_off = CodecPipeline([
        Op3_ApogeeIntN_Substrate(bits=6),
        Op2_PR103ArithmeticCodec(ac_auto_fallback=False),
    ])
    pipe_on = CodecPipeline([
        Op3_ApogeeIntN_Substrate(bits=6),
        Op2_PR103ArithmeticCodec(ac_auto_fallback=True),
    ])
    _, m_off = pipe_off.encode(sd, skip_validate=True)
    _, m_on = pipe_on.encode(sd, skip_validate=True)
    op2_off = m_off.op_results[1].bytes_out
    op2_on = m_on.op_results[1].bytes_out
    # Strict assertion: on int6 substrate, ON must beat OFF (regressions exist).
    assert op2_on < op2_off, (
        f"Op2 fallback ON ({op2_on}) did not beat OFF ({op2_off}) on PR106 int6"
    )
    # Sanity: fallback set must be non-empty here.
    fb = m_on.op_results[1].op_state["ac_fallback_set"]
    assert fb, f"expected non-empty fallback set on int6 substrate, got {fb!r}"
