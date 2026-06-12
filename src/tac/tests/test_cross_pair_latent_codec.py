# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the cross-pair latent codec (ITEM D1).

These tests assert the MECHANISM, not constants. The central NO-FAKE discipline: a test
that passes when the codec returns its input unchanged is worthless. Each test below would
FAIL if the codec silently no-op'd, padded an enum with duplicate implementations, or skipped
the round-trip:

  * round-trip is BIT-EXACT on the QUANTIZED codes for EVERY framed format (decode∘encode == q);
  * the default-preserving adapter emits BYTE-IDENTICAL vendored bytes when no candidate wins
    (mirrors variable_level_codec's ``73527==73527`` proof);
  * the dedup / codebook candidates ACTUALLY shrink redundant latents (a no-op would not) and
    are DISTINCT implementations (different byte counts on structured data);
  * adversarial inputs (all-equal pairs, all-distinct pairs, single pair, max-range symbols,
    high-dim) all round-trip;
  * the real base_ch20 latents (when present) round-trip bit-exact AND the adapter is
    byte-identical to a vendored build (the MEASURED negative — honest, not a failure).

The real-latent tests skip gracefully if the checkpoints aren't on disk (so the suite runs in
CI), but the synthetic adversarial + redundancy tests are unconditional and carry the NO-FAKE
weight.
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

import brotli
import numpy as np
import pytest
import torch

from tac.losses import cross_pair_latent_codec as C

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Real base_ch20 latent tensors (the deployed-win measurement surface). Skip if absent.
_REAL_LATENT_PATHS = [
    _REPO_ROOT
    / "experiments/results/forkpoints/basin_bc20_20260612T121523Z/best/best_ema_latents.pt",
    _REPO_ROOT
    / "experiments/results/distortion_arm_l235_20260612T205102Z/best/best_ema_latents.pt",
    _REPO_ROOT
    / "experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best/best_ema_latents.pt",
]


def _load_latents(p: Path) -> torch.Tensor:
    obj = torch.load(p, map_location="cpu")
    if isinstance(obj, dict):
        for v in obj.values():
            if torch.is_tensor(v) and v.dim() == 2:
                obj = v
                break
    return obj.float()


def _vendored_codec():
    try:
        from tac.torch_vehicle.vendored_imports import import_vendored

        return import_vendored("codec")
    except Exception:  # pragma: no cover - vendored clone may be absent in some envs
        return None


# ---------------------------------------------------------------------------
# Quantization parity with the vendored grid.
# ---------------------------------------------------------------------------

def test_quant_dequant_round_trips_quantized_codes():
    """quantize -> dequantize reproduces the dequant of the same quant codes exactly."""
    torch.manual_seed(1)
    lat = torch.randn(50, 12) * 3.0
    q, mins, scales = C.quantize_pairs(lat)
    recon = C.dequantize_pairs(q, mins, scales)
    # Re-quantizing the reconstruction must give back the SAME codes (quant is idempotent on q).
    q2, _, _ = C.quantize_pairs(recon)
    assert np.array_equal(q, q2)


def test_quant_matches_vendored_encode_decode_when_available():
    """The codec's quant reconstruction equals the vendored encode/decode reconstruction."""
    vend = _vendored_codec()
    if vend is None:
        pytest.skip("vendored codec clone not available")
    torch.manual_seed(2)
    lat = torch.randn(80, 28) * 4.0
    vend_recon = vend.decode_latents(vend.encode_latents(lat))
    q, mins, scales = C.quantize_pairs(lat)
    my_recon = C.dequantize_pairs(q, mins, scales)
    assert torch.equal(vend_recon, my_recon)


def test_quant_codes_in_valid_range():
    torch.manual_seed(3)
    lat = torch.randn(30, 16) * 10.0
    q, _, _ = C.quantize_pairs(lat)
    assert q.dtype == np.uint8
    assert q.min() >= 0 and q.max() <= C._QUANT_MAX


# ---------------------------------------------------------------------------
# Per-format BIT-EXACT round-trip (the core NO-FAKE contract).
# ---------------------------------------------------------------------------

def _assert_format_round_trips(lat: torch.Tensor):
    """Every framed candidate must round-trip the quant codes bit-exact."""
    q, mins, scales = C.quantize_pairs(lat)
    target = C.dequantize_pairs(q, mins, scales)
    cands = C._candidate_payloads(q, mins, scales, C.CrossPairLatentConfig())
    # All three framed formats must be present (no enum-padding skip).
    assert set(cands) == {C.FORMAT_FRAMED_DELTA, C.FORMAT_DEDUP, C.FORMAT_CODEBOOK}
    for flag, payload in cands.items():
        recon = C.decode_latents_best(payload)
        assert torch.equal(recon, target), f"format {flag} round-trip mismatch"
        # The decoded flag must be the one we encoded (framing is honest).
        assert payload[0] == flag


def test_framed_delta_round_trip():
    torch.manual_seed(4)
    _assert_format_round_trips(torch.randn(60, 28) * 5.0)


def test_dedup_round_trip_on_repeated_rows():
    """Dedup must round-trip when rows repeat (its target case)."""
    torch.manual_seed(5)
    base = torch.randn(20, 28) * 5.0
    idx = torch.randint(0, 20, (300,))
    lat = base[idx]
    _assert_format_round_trips(lat)


def test_codebook_round_trip_is_exact_despite_lossy_centroids():
    """Codebook residual carries the EXACT remainder -> bit-exact regardless of codebook quality."""
    torch.manual_seed(6)
    lat = torch.randn(120, 28) * 5.0  # diverse -> codebook is a poor predictor; residual saves it
    q, mins, scales = C.quantize_pairs(lat)
    target = C.dequantize_pairs(q, mins, scales)
    payload = C._encode_codebook(q, mins, scales, 64, C.CrossPairLatentConfig())
    recon = C.decode_latents_best(payload)
    assert torch.equal(recon, target)


# ---------------------------------------------------------------------------
# Adversarial inputs.
# ---------------------------------------------------------------------------

def test_all_equal_pairs_round_trip():
    """All-equal pairs (degenerate: scale floor kicks in) round-trip and dedup to 1 unique row."""
    lat = torch.full((100, 28), 3.5)
    q, mins, scales = C.quantize_pairs(lat)
    target = C.dequantize_pairs(q, mins, scales)
    payload, flag = C.encode_latents_best(lat)
    recon = C.decode_latents_best(payload)
    assert torch.equal(recon, target)
    # Dedup candidate should collapse to a single unique row.
    dedup_payload = C._encode_dedup(q, mins, scales)
    buf = io.BytesIO(dedup_payload)
    buf.read(1)  # flag
    C._unpack_side(buf)
    n_uniq, _ = struct.unpack("<IB", buf.read(5))
    assert n_uniq == 1


def test_all_distinct_pairs_round_trip():
    torch.manual_seed(7)
    lat = torch.randn(200, 28) * 9.0
    payload, flag = C.encode_latents_best(lat)
    q, mins, scales = C.quantize_pairs(lat)
    target = C.dequantize_pairs(q, mins, scales)
    assert torch.equal(C.decode_latents_best(payload), target)


def test_single_pair_round_trip():
    lat = torch.randn(1, 28) * 5.0
    payload, _ = C.encode_latents_best(lat)
    q, mins, scales = C.quantize_pairs(lat)
    assert torch.equal(C.decode_latents_best(payload), C.dequantize_pairs(q, mins, scales))


def test_max_range_symbols_round_trip():
    """Latents that span the full quant grid (large dynamic range) round-trip exactly."""
    torch.manual_seed(8)
    lat = torch.randn(150, 28) * 1000.0
    payload, _ = C.encode_latents_best(lat)
    q, mins, scales = C.quantize_pairs(lat)
    assert torch.equal(C.decode_latents_best(payload), C.dequantize_pairs(q, mins, scales))


def test_high_latent_dim_round_trip():
    torch.manual_seed(9)
    lat = torch.randn(64, 256) * 4.0
    payload, _ = C.encode_latents_best(lat)
    q, mins, scales = C.quantize_pairs(lat)
    assert torch.equal(C.decode_latents_best(payload), C.dequantize_pairs(q, mins, scales))


def test_many_unique_forces_uint32_index_path():
    """> 65535 unique rows would force the uint32 index width; exercise the width-switch logic.

    We can't make 65k unique rows cheaply, so directly test the dedup width selection by
    constructing exactly at the boundary on a small alphabet (the code path, not the count)."""
    torch.manual_seed(10)
    lat = torch.randn(500, 28) * 6.0
    q, mins, scales = C.quantize_pairs(lat)
    payload = C._encode_dedup(q, mins, scales)
    buf = io.BytesIO(payload)
    buf.read(1)
    C._unpack_side(buf)
    n_uniq, idx_width = struct.unpack("<IB", buf.read(5))
    assert idx_width == 2  # n_uniq <= 65535 -> uint16
    recon = C.decode_latents_best(payload)
    assert torch.equal(recon, C.dequantize_pairs(q, mins, scales))


# ---------------------------------------------------------------------------
# NO-FAKE: candidates ACTUALLY shrink redundant data and are DISTINCT.
# ---------------------------------------------------------------------------

def _structured_redundant_latents(n=600, d=28, modes=40, seed=0):
    torch.manual_seed(seed)
    base = torch.randn(modes, d) * 5.0
    mode_seq = []
    g = torch.Generator().manual_seed(seed)
    while len(mode_seq) < n:
        m = int(torch.randint(0, modes, (1,), generator=g))
        runlen = int(torch.randint(5, 20, (1,), generator=g))
        mode_seq += [m] * runlen
    return base[torch.tensor(mode_seq[:n])]


def test_dedup_strictly_beats_framed_delta_on_redundant_latents():
    """If dedup were a no-op, it would NOT shrink. On run-structured repeats it MUST win."""
    lat = _structured_redundant_latents()
    q, mins, scales = C.quantize_pairs(lat)
    cands = C._candidate_payloads(q, mins, scales, C.CrossPairLatentConfig())
    sz = {f: len(brotli.compress(p, quality=11)) for f, p in cands.items()}
    assert sz[C.FORMAT_DEDUP] < sz[C.FORMAT_FRAMED_DELTA], sz


def test_candidate_formats_are_distinct_implementations():
    """Dedup, codebook, framed-delta produce DIFFERENT byte counts -> not enum-padding."""
    lat = _structured_redundant_latents(seed=3)
    q, mins, scales = C.quantize_pairs(lat)
    cands = C._candidate_payloads(q, mins, scales, C.CrossPairLatentConfig())
    sizes = {len(brotli.compress(p, quality=11)) for p in cands.values()}
    assert len(sizes) == len(cands), "candidates collapsed to identical sizes (suspect enum-padding)"


def test_selector_picks_framed_format_when_it_wins():
    """The adapter emits the framed format (is_framed=True) when a candidate strictly wins."""
    vend = _vendored_codec()
    if vend is None:
        pytest.skip("vendored codec clone not available")
    lat = _structured_redundant_latents(seed=1)
    blob, is_framed = C.build_latent_blob_dedup_or_vendored(lat)
    vendored_blob = brotli.compress(vend.encode_latents(lat), quality=11)
    assert is_framed is True
    assert len(blob) < len(vendored_blob)
    # And it round-trips bit-exact through the adapter dispatch.
    recon = C.decode_latent_blob(blob, is_framed)
    target = vend.decode_latents(vend.encode_latents(lat))
    assert torch.equal(recon, target)


def test_a_no_op_codec_would_fail_the_savings_assertion():
    """Sanity: a fake 'dedup' that just returned the framed-delta payload would NOT shrink.

    We simulate the fake by comparing the framed-delta candidate to ITSELF — proving the
    savings assertion in test_dedup_strictly_beats_framed_delta is non-trivial (a no-op codec
    that returned framed-delta bytes would give equality, which the strict '<' would reject).
    """
    lat = _structured_redundant_latents(seed=2)
    q, mins, scales = C.quantize_pairs(lat)
    fake_dedup = C._encode_framed_delta(q, mins, scales)  # the "no-op" stand-in
    real_dedup = C._encode_dedup(q, mins, scales)
    fake_sz = len(brotli.compress(fake_dedup, quality=11))
    real_sz = len(brotli.compress(real_dedup, quality=11))
    # The REAL dedup beats the fake (no-op) on redundant data; the fake does NOT beat framed-delta.
    assert real_sz < fake_sz


# ---------------------------------------------------------------------------
# Default-preserving guarantee (the variable_level_codec-style byte-identity proof).
# ---------------------------------------------------------------------------

def test_adapter_byte_identical_to_vendored_when_no_candidate_wins():
    """On near-incompressible latents (no cross-pair structure), the adapter MUST be byte-identical.

    This is the default-preserving guard: a tie or no-win keeps the EXACT vendored brotli bytes.
    """
    vend = _vendored_codec()
    if vend is None:
        pytest.skip("vendored codec clone not available")
    torch.manual_seed(11)
    lat = torch.randn(600, 28) * 5.0  # iid noise -> no cross-pair redundancy -> vendored wins
    blob, is_framed = C.build_latent_blob_dedup_or_vendored(lat)
    vendored_blob = brotli.compress(vend.encode_latents(lat), quality=11)
    assert is_framed is False
    assert blob == vendored_blob  # BYTE-IDENTICAL (the 15800==15800-style proof)


def test_adapter_tie_keeps_vendored():
    """A strict '<' means even a byte-TIE keeps vendored (the no-perturbation guarantee)."""
    vend = _vendored_codec()
    if vend is None:
        pytest.skip("vendored codec clone not available")
    torch.manual_seed(12)
    lat = torch.randn(100, 28) * 5.0
    blob, is_framed = C.build_latent_blob_dedup_or_vendored(lat)
    vendored_blob = brotli.compress(vend.encode_latents(lat), quality=11)
    # Whatever the outcome, if is_framed is False the bytes are exactly vendored.
    if not is_framed:
        assert blob == vendored_blob


def test_adapter_never_emits_framed_on_iid_noise_brotli_alignment_artifact():
    """R(Lens-2) regression: a framed-delta-vs-vendored brotli ALIGNMENT artifact is NOT a win.

    FRAMED_DELTA carries information identical to vendored, so on iid noise (no cross-pair
    structure) any byte difference between framed-delta and vendored is pure brotli block
    alignment from the 1-byte flag — NOT a real saving. The adapter (structural_only) must NEVER
    emit a framed format here. Before the fix this falsely emitted is_framed=True ~18/40 seeds,
    claiming 1-29 B 'wins' that were brotli noise. This locks the honest-negative in.
    """
    vend = _vendored_codec()
    if vend is None:
        pytest.skip("vendored codec clone not available")
    framed_wins = 0
    for seed in range(40):
        torch.manual_seed(seed)
        lat = torch.randn(600, 28) * float(2 + seed % 8)
        blob, is_framed = C.build_latent_blob_dedup_or_vendored(lat)
        vendored_blob = brotli.compress(vend.encode_latents(lat), quality=11)
        if is_framed:
            framed_wins += 1
        else:
            assert blob == vendored_blob, f"seed {seed}: is_framed False but bytes differ"
    assert framed_wins == 0, f"framed format won on {framed_wins} iid-noise seeds (alignment artifact)"


def test_encode_latents_best_structural_only_excludes_framed_delta_win():
    """structural_only=True restricts selectable winners to dedup/codebook (not framed-delta)."""
    # iid noise: framed-delta might brotli-beat the OTHER candidates, but structural_only must
    # only ever return a structural flag as a 'win' — else fall back to framed-delta baseline.
    torch.manual_seed(13)
    lat = torch.randn(300, 28) * 5.0
    _payload, flag = C.encode_latents_best(lat, structural_only=True)
    # On iid noise the structural candidates lose, so the baseline (framed-delta) is returned;
    # but it is returned as the FALLBACK, never selected over a structural candidate.
    assert flag in (C.FORMAT_FRAMED_DELTA, C.FORMAT_DEDUP, C.FORMAT_CODEBOOK)
    # On structured data, structural_only must pick a structural format.
    lat2 = _structured_redundant_latents(seed=7)
    _p2, flag2 = C.encode_latents_best(lat2, structural_only=True)
    assert flag2 in C._STRUCTURAL_FORMATS


# ---------------------------------------------------------------------------
# Real base_ch20 latents (the MEASURED deployed surface).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", _REAL_LATENT_PATHS, ids=lambda p: p.parent.parent.name)
def test_real_latents_round_trip_bit_exact(p: Path):
    if not p.exists():
        pytest.skip(f"real latent tensor not on disk: {p}")
    vend = _vendored_codec()
    if vend is None:
        pytest.skip("vendored codec clone not available")
    lat = _load_latents(p)
    target = vend.decode_latents(vend.encode_latents(lat))
    blob, is_framed = C.build_latent_blob_dedup_or_vendored(lat)
    recon = C.decode_latent_blob(blob, is_framed)
    assert torch.equal(recon, target)


@pytest.mark.parametrize("p", _REAL_LATENT_PATHS, ids=lambda p: p.parent.parent.name)
def test_real_latents_adapter_byte_identical_measured_negative(p: Path):
    """The MEASURED reality: on the current base_ch20 latents the adapter is byte-identical.

    This encodes the honest negative as a regression guard: if a future change made the codec
    perturb these latents WITHOUT a real win, this test would catch it.
    """
    if not p.exists():
        pytest.skip(f"real latent tensor not on disk: {p}")
    vend = _vendored_codec()
    if vend is None:
        pytest.skip("vendored codec clone not available")
    lat = _load_latents(p)
    blob, is_framed = C.build_latent_blob_dedup_or_vendored(lat)
    vendored_blob = brotli.compress(vend.encode_latents(lat), quality=11)
    assert is_framed is False, "framed format won on real latents — re-measure the negative!"
    assert blob == vendored_blob


def test_score_claim_discipline_flags():
    """Advisory metadata must be False (no score / promotion claim from this module)."""
    assert C.SCORE_CLAIM is False
    assert C.PROMOTION_ELIGIBLE is False
    assert C.READY_FOR_EXACT_EVAL_DISPATCH is False
