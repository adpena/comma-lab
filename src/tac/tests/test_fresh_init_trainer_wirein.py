"""Static and pure wire-in guards for FreSh in the live MLX trainer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tac.witness_init.fresh_frequency_shift import (
    deterministic_first_layer_bias_candidates,
)

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("fresh_wirein_trainer", _TRAINER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_finer_bias_stream_is_single_sourced_through_fresh_helper() -> None:
    module = _load_trainer()
    for seed in (0, 7, 20260712):
        for k in (0.1, 3.0):
            actual = module._finer_bias_init_values(seed, k, 17)
            expected = deterministic_first_layer_bias_candidates((k,), 17, seed=seed)[0]
            assert actual.dtype == np.float32
            assert np.array_equal(actual, expected)
    source = _TRAINER.read_text()
    finer_body = source[source.index("def _finer_bias_init_values"):source.index(
        "def _logit_adjust_classes_mask"
    )]
    assert "deterministic_first_layer_bias_candidates" in finer_body
    assert "default_rng" not in finer_body


def test_fresh_selector_runs_after_siren_and_before_structured_prefit() -> None:
    source = _TRAINER.read_text()
    siren = source.index("apply_siren_init(model")
    fresh = source.index("run_fresh_initialization_sweep(")
    structured = source.index("if args.structured_init:", fresh)
    assert siren < fresh < structured
    assert "score_fresh_committed_state(" in source[structured:structured + 18000]


def test_fresh_is_checkpointed_registered_and_exposed_in_result() -> None:
    source = _TRAINER.read_text()
    assert '_resume_registry.register("fresh_init"' in source
    assert "fresh_checkpoint_cfg_arrays" in source
    assert '"fresh_init": _fresh_state.result_dict()' in source
    assert "fresh_init_receipt.json" in source
    assert "fresh_init_post_structured_receipt.json" in source
    assert "fresh_init_blocker.json" in source
    assert "matched_fresh_arm_config(args)" in source
    assert "fresh_training_target_sha256" in source
    assert "fresh_init_scorer_accounting(" in source
    assert '"total_init_seconds_to_epoch0"' in source


def test_parser_defaults_are_off_and_companion_values_are_typed() -> None:
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser

    parser = build_real_trainer_parser()
    args = parser.parse_args(
        ["--out-dir", "experiments/results/test_fresh_parser_defaults"]
    )
    assert args.fresh_init is False
    assert args.fresh_spectrum_size == 64
    assert args.fresh_sample_pairs == 10
    assert args.fresh_reference_freq_along == pytest.approx(8.0)
    assert args.fresh_tangent_deficit == pytest.approx(3.2)
    assert args.fresh_bias_k_min == pytest.approx(0.0)
    assert args.fresh_bias_k_max == pytest.approx(3.0)
    assert args.fresh_bias_k_step == pytest.approx(0.1)
