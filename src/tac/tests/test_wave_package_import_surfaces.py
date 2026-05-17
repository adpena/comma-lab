# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib


def test_quantization_wave_import_surface_is_not_dangling() -> None:
    mod = importlib.import_module("tac.quantization_wave")

    assert set(mod.IMPLEMENTED_MODULES) == {
        "apple_neural_engine_export",
        "awq_activation_aware_quantization",
        "balle_hyperprior_bolton",
        "entropy_coding_archive_primitives",
        "fp4_quantization_wave",
        "fp8_quantization_wave",
        "gguf_style_per_tensor_mixed_bit",
        "gptq_post_training_quantization",
        "int4_int8_mixed_bit",
        "mlx_inference_path",
        "sparse_weights_with_quant",
        "vq_codebook_quantization",
    }
    assert mod.DEFERRED_MODULES == ()
    assert "GPTQStyleQuantizer" in mod.__all__
    assert "DEFERRED_RATIONALE" in mod.__all__


def test_training_curriculum_import_surface_is_not_dangling() -> None:
    mod = importlib.import_module("tac.training_curriculum")

    assert set(mod.IMPLEMENTED_MODULES) == {
        "a1_pattern_inflate_time_bias_correction",
        "early_stopping_with_resume",
        "model_soup_averaging",
        "multi_stage_curriculum",
        "pause_and_diagnose",
        "pause_distill_resume",
        "pause_quantize_finetune",
        "pause_to_swap_loss",
        "swa_polyak_averaging",
    }
    assert mod.DEFERRED_MODULES == ()
    assert "DistillationConfig" in mod.__all__
    assert "DEFERRED_RATIONALE" in mod.__all__


def test_contest_exploits_import_surface_marks_only_missing_chroma_anchor() -> None:
    mod = importlib.import_module("tac.contest_exploits")

    assert set(mod.IMPLEMENTED_MODULES) == {
        "cpu_cuda_gap_exploit",
        "deterministic_scorer_exploit",
        "kaggle_ensemble_pattern",
        "pair_index_lookup_table",
        "per_class_chroma_anchor",
        "per_frame_hardcoded_params",
        "precomputed_inference_outputs",
        "rate_weight_exploit",
        "test_time_adaptation",
        "video_known_structure_priors",
    }
    assert mod.DEFERRED_MODULES == ()
    assert "PerFrameHardcodedParams" in mod.__all__
    assert "DEFERRED_RATIONALE" in mod.__all__
