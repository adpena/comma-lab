# SPDX-License-Identifier: MIT
"""Compliance regressions for the VQ-VAE K-sweep dispatch surface."""

from __future__ import annotations

from pathlib import Path

RECIPE = Path(
    ".omx/operator_authorize_recipes/"
    "substrate_vq_vae_k_sweep_modal_a10g_diagnostic_dispatch.yaml"
)
TRAINER = Path("experiments/train_substrate_vq_vae.py")


def test_k_sweep_recipe_does_not_claim_modal_training_is_contest_cuda() -> None:
    text = RECIPE.read_text(encoding="utf-8")

    assert "Single-axis [contest-CUDA] acceptable for empirical anchor" not in text
    assert "AUTH_EVAL_DEVICE=cpu" in text
    assert "MODAL_AUTH_EVAL_ADVISORY_ONLY=1" in text
    assert "score_claim=false" in text
    assert "separate claimed exact-CUDA eval" in text


def test_k_sweep_rate_matrix_is_labelled_as_unimplemented_floor() -> None:
    text = RECIPE.read_text(encoding="utf-8")

    assert "fixed-int16 VQV1 reconstruction/quality diagnostic" in text
    assert "not a" in text
    assert "proof of the Wave 2A bit-packed Pareto frontier" in text
    assert "K-dependent archive grammar" in text
    assert "vq_v1_k_dependent_entropy_packed_archive" in text
    assert "analytical bit-packed floor, NOT current archive bytes" in text
    assert "byte impact must be measured from archive.zip/0.bin" in text
    assert "bit-packer implementation evidence" in text


def test_vq_vae_trainer_keeps_cpu_auth_eval_out_of_cuda_claim_fields() -> None:
    text = TRAINER.read_text(encoding="utf-8")

    assert 'os.environ.get("AUTH_EVAL_DEVICE", "").strip()' in text
    assert 'contest_auth_eval_{auth_eval_device_name}.json' in text
    assert "return_non_cuda_result=True" in text
    assert '"auth_eval_result": auth_eval_result' in text
    assert '"auth_eval_custody_complete": auth_eval_custody_complete' in text
    assert '"score_claim": auth_eval_score_claim_valid' in text
    assert "contest-CUDA score demoted because auth-eval " in text
    assert "custody is incomplete" in text


def test_vq_vae_trainer_binds_score_custody_to_evaluated_archive_zip() -> None:
    text = TRAINER.read_text(encoding="utf-8")

    assert "payload_bin_sha = _sha256_bytes(bin_bytes)" in text
    assert "archive_zip_bytes = archive_zip_path.read_bytes()" in text
    assert "archive_sha = _sha256_bytes(archive_zip_bytes)" in text
    assert '"payload_bin_sha256": payload_bin_sha' in text
    assert '"archive_sha256": archive_sha' in text
    assert "auth_eval_archive_size_matches" in text
    assert "auth_eval_archive_sha256_matches" in text
    assert "auth_eval_runtime_tree_sha256_matches" in text
    assert "runtime_tree_sha256" in text
