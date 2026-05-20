# SPDX-License-Identifier: MIT
"""Regression checks for the VQ-VAE remote lane driver."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path("scripts/remote_lane_substrate_vq_vae.sh")


def test_vq_vae_remote_driver_threads_k_sweep_env_to_trainer() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'VQ_VAE_CODEBOOK_SIZE="${VQ_VAE_CODEBOOK_SIZE:-}"' in text
    assert 'VQ_VAE_ALPHA_RATE="${VQ_VAE_ALPHA_RATE:-}"' in text
    assert "VQ_VAE_CODEBOOK_SIZE is required by the K-sweep recipe" in text
    assert "VQ_VAE_ALPHA_RATE is required by the K-sweep recipe" in text
    assert "refusing phantom default K" in text
    assert "refusing phantom default alpha" in text
    assert "'codebook_size': $VQ_VAE_CODEBOOK_SIZE" in text
    assert "'alpha_rate': $VQ_VAE_ALPHA_RATE" in text
    assert '--codebook-size "$VQ_VAE_CODEBOOK_SIZE"' in text
    assert '--alpha-rate "$VQ_VAE_ALPHA_RATE"' in text


def test_vq_vae_remote_driver_completion_marker_uses_actual_auth_eval_axis() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'AUTH_EVAL_DEVICE_RESOLVED="${AUTH_EVAL_DEVICE:-$VQ_VAE_DEVICE}"' in text
    assert 'contest_auth_eval_${AUTH_EVAL_DEVICE_RESOLVED}.json' in text
    assert 'ARCHIVE_PATH="$OUTPUT_DIR/archive.zip"' in text
    assert 'ARCHIVE_PATH="$OUTPUT_DIR/0.bin"' not in text
    assert 'LANE_VQ_VAE_DONE [contest-CUDA]' in text
    assert 'LANE_VQ_VAE_DONE [diagnostic-auth-eval]' in text
    assert "score_claim=false promotion_eligible=false" in text


def test_vq_vae_remote_provenance_is_diagnostic_not_frontier_claim() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "fixed_int16_vqv1_quality_probe_no_frontier_authority" in text
    assert "'predicted_band': None" in text
    assert "'predicted_basis': 'none_for_fixed_int16_diagnostic'" in text
    assert "'class_shift_followup': 'vq_v1_k_dependent_entropy_packed_archive'" in text
    assert "'score_claim': False" in text
    assert "'rank_or_kill_eligible': False" in text
