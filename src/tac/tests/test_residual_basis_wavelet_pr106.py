from __future__ import annotations

import numpy as np
import pytest

from tac.residual_basis import (
    WaveletResidualError,
    compute_wavelet_residual_stats,
    decompose_frame_to_bands,
    load_decoded_raw_frames,
    reconstruct_frame_from_bands,
)


def test_wavelet_residual_stats_are_research_signal_only() -> None:
    rng = np.random.default_rng(20260511)
    frames = rng.integers(0, 256, size=(3, 32, 32, 3), dtype=np.uint8)

    result = compute_wavelet_residual_stats(frames, wavelet="haar", levels=2)

    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.evidence_grade == "research_signal"
    assert result.n_frames == 3
    assert result.height == 32
    assert result.width == 32
    assert {s.band_name for s in result.per_band_stats} == {"LL", "LH", "HL", "HH"}
    assert {s.level for s in result.per_band_stats} == {1, 2}
    assert all(s.n_coefficients > 0 for s in result.per_band_stats)


def test_wavelet_decompose_reconstruct_round_trips_power_of_two_frame() -> None:
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    bands = decompose_frame_to_bands(frame, wavelet="haar", levels=2)
    recovered = reconstruct_frame_from_bands(bands, wavelet="haar")

    assert recovered.shape == frame.shape
    np.testing.assert_allclose(recovered, frame.astype(np.float32), atol=1e-4)


def test_wavelet_residual_rejects_bad_shape() -> None:
    with pytest.raises(WaveletResidualError, match="expected"):
        compute_wavelet_residual_stats(np.zeros((32, 32, 3), dtype=np.uint8))


def test_load_decoded_raw_frames_reads_bounded_prefix(tmp_path) -> None:
    raw = tmp_path / "0.raw"
    frames = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    raw.write_bytes(frames.tobytes())

    out = load_decoded_raw_frames(raw, height=4, width=5, max_frames=1)

    assert out.shape == (1, 4, 5, 3)
    np.testing.assert_array_equal(out, frames[:1])
