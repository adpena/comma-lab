# SPDX-License-Identifier: MIT
"""Tests for the ATW V2-1 Faiss-PQ disambiguator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import probe_atw_v2_1_faiss_pq_disambiguator as probe  # noqa: E402


def test_region_means_and_top_region_selection() -> None:
    probs = np.zeros((5, 4, 4), dtype=np.float32)
    probs[0, :2, :2] = 1.0
    probs[1, :2, 2:] = 1.0
    probs[2, 2:, :2] = 1.0
    probs[:, 2:, 2:] = 0.2

    means = probe._softmax_region_means(probs, grid_side=2)
    assert means.shape == (4, 5)
    np.testing.assert_allclose(means[0], [1, 0, 0, 0, 0])
    np.testing.assert_allclose(means[3], [0.2, 0.2, 0.2, 0.2, 0.2])
    # The uniform region has zero entropy deficit, so it should not be selected.
    assert probe.select_region_indices(means, top_k=2) == [0, 1]


def test_compute_pq_mi_verdict_detects_correlated_side_info() -> None:
    latent = bytes([0, 0, 0, 0, 255, 255, 255, 255])
    verdict = probe.compute_pq_mi_verdict(
        latent_stream=latent,
        per_pair_symbols=[10, 20],
        symbols_per_pair=4,
        threshold=0.5,
    )
    assert verdict.verdict == "MEANINGFUL_CONDITIONING"
    assert verdict.mutual_information_bits >= 0.99
    assert verdict.num_unique_side_info_symbols == 2


def test_missing_faiss_payload_is_fail_closed(tmp_path: Path) -> None:
    payload = probe.missing_faiss_payload(
        output_dir=tmp_path,
        error=ImportError("No module named 'faiss'"),
    )
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["phase2_status"] == "dependency_blocked_faiss_cpu_missing"
    assert "faiss_cpu_dependency_missing" in payload["result_review_blockers"]


def test_build_probe_payload_with_fake_pq_backend(monkeypatch, tmp_path: Path) -> None:
    class FakeCodebook:
        d = 5

        def sa_code_size(self) -> int:
            return 1

    def fake_build(training_vectors, *, nlist, m_subq, nbits, seed):
        assert training_vectors.shape[1] == 5
        return FakeCodebook()

    def fake_encode(softmax_per_pair, codebook):
        # One deterministic code byte per selected region, correlated with class 0.
        return bytes(int(row[0] > 0.5) for row in softmax_per_pair)

    def fake_decode(encoded_bytes, codebook, *, n_regions, softmax_dim=5):
        return np.zeros((n_regions, softmax_dim), dtype=np.float32)

    monkeypatch.setattr(probe, "build_pq_codebook", fake_build)
    monkeypatch.setattr(probe, "serialize_codebook", lambda codebook: b"fake-codebook")
    monkeypatch.setattr(probe, "encode_per_region_histogram", fake_encode)
    monkeypatch.setattr(probe, "decode_per_region_histogram", fake_decode)

    n_pairs = 8
    softmax16 = np.zeros((n_pairs, 16, 5), dtype=np.float32)
    softmax256 = np.zeros((n_pairs, 256, 5), dtype=np.float32)
    latent = bytearray()
    for pair_idx in range(n_pairs):
        cls = pair_idx % 2
        softmax16[pair_idx, :, cls] = 1.0
        softmax256[pair_idx, :, cls] = 1.0
        latent.extend([0 if cls == 0 else 255] * 4)

    payload = probe.build_probe_payload(
        latent_stream=bytes(latent),
        softmax_by_region_count={16: softmax16, 256: softmax256},
        output_dir=tmp_path,
        softmax_provenance={"fixture": "fake_pq_backend"},
    )

    assert payload["schema"] == "atw_v2_1_faiss_pq_disambiguator_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert len(payload["variants"]) == 3
    assert payload["best_variant"]["verdict"] == "MEANINGFUL_CONDITIONING"
    assert payload["phase2_status"] == (
        "pq_variant_meaningful_requires_new_d4_and_wave_n_plus_1_council"
    )
    for row in payload["variants"]:
        assert Path(row["codeword_stream_path"]).name.endswith("_pq_stream.bin")
        assert row["verdict"]["mutual_information_bits"] >= 0.99
