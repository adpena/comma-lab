"""Pure regression tests for the FreSh trainer/resume contract."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from tac.witness_init.fresh_trainer_contract import (
    FRESH_RESUME_PREFIX,
    FreShInitState,
    fresh_checkpoint_cfg_arrays,
    fresh_state_arrays,
    fresh_training_target_sha256,
    load_checkpoint_cfg,
    matched_fresh_arm_config,
    restore_fresh_checkpoint_before_features,
    restore_fresh_state_from_cfg,
    validate_fresh_init_args,
)


def _args(**changes: object) -> Namespace:
    values: dict[str, object] = {
        "fresh_init": True,
        "fresh_init_control": False,
        "self_orient": True,
        "activation": "hosc",
        "siren_init": True,
        "n_dir_freqs": 4,
        "finer_bias_init": False,
        "freeze_decoder_fit_codes": None,
        "seed_islands": False,
        "residual_mode": False,
        "lane_render_band": False,
        "lane_band_start_epoch": 500,
        "render_aa": "none",
        "fresh_spectrum_size": 64,
        "fresh_sample_pairs": 10,
        "freq_along": 8.0,
        "freq_across": 32.0,
        "fresh_reference_freq_along": 8.0,
        "fresh_tangent_deficit": 3.2,
        "fresh_bias_k_min": 0.0,
        "fresh_bias_k_max": 3.0,
        "fresh_bias_k_step": 0.1,
    }
    values.update(changes)
    return Namespace(**values)


def _applied_state() -> FreShInitState:
    return FreShInitState(
        enabled=True,
        applied=True,
        candidate_index=7,
        selected_freq_along=8.0 * np.sqrt(3.2),
        selected_bias_k=0.3,
        selected_mean_distance=1.25,
        post_structured_mean_distance=1.5,
        init_seconds=2.75,
        selection_receipt_sha256="a" * 64,
    )


def test_validation_is_noop_off_and_counts_exact_default_grid() -> None:
    assert validate_fresh_init_args(
        _args(fresh_init=False, fresh_init_control=False, self_orient=False)
    ) == 0
    # current=reference=8 stable-deduplicates, leaving {8, 8*sqrt(3.2), 25.6}.
    assert validate_fresh_init_args(_args()) == 3 * 31
    assert validate_fresh_init_args(_args(fresh_init=False, fresh_init_control=True)) == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"self_orient": False}, "requires --self-orient"),
        ({"activation": "relu"}, "periodic --activation"),
        ({"siren_init": False}, "requires --siren-init"),
        ({"n_dir_freqs": 0}, "--n-dir-freqs > 0"),
        ({"finer_bias_init": True}, "conflicts with --finer-bias-init"),
        ({"freeze_decoder_fit_codes": "decoder.npz"}, "conflicts with --freeze-decoder"),
        ({"render_aa": "supersample"}, "not supersample"),
        ({"seed_islands": True}, "no-seed-islands"),
        ({"residual_mode": True}, "residual mode OFF"),
        ({"lane_render_band": True, "lane_band_start_epoch": 1}, "start after epoch 0"),
        ({"fresh_tangent_deficit": 1.0}, "must be > 1"),
        ({"fresh_spectrum_size": 0}, "positive integer"),
        ({"fresh_bias_k_min": 0.1}, "must be 0"),
        ({"fresh_bias_k_step": 0.07}, "divide"),
    ],
)
def test_invalid_typed_compositions_fail_closed(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_fresh_init_args(_args(**changes))


def test_ipe_input_basis_is_supported_for_v9_composition() -> None:
    assert validate_fresh_init_args(_args(render_aa="ipe")) == 3 * 31


def test_matched_config_excludes_only_arm_identity_and_output_path() -> None:
    treatment_payload, treatment_sha = matched_fresh_arm_config(
        _args(out_dir="treatment", fresh_init=True, fresh_init_control=False)
    )
    control_payload, control_sha = matched_fresh_arm_config(
        _args(out_dir="control", fresh_init=False, fresh_init_control=True)
    )
    assert treatment_payload == control_payload
    assert treatment_sha == control_sha
    assert "out_dir" not in treatment_payload
    assert "fresh_init" not in treatment_payload
    assert "fresh_init_control" not in treatment_payload
    changed_payload, changed_sha = matched_fresh_arm_config(
        _args(out_dir="control", fresh_init=False, fresh_init_control=True, seed=8)
    )
    assert changed_payload != control_payload
    assert changed_sha != control_sha


def test_matched_config_refuses_unserializable_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="unsupported type"):
        matched_fresh_arm_config(_args(callback=object()))
    with pytest.raises(ValueError, match="must be finite"):
        matched_fresh_arm_config(_args(freq_along=float("nan")))


def test_training_target_authority_hash_binds_every_active_array() -> None:
    fields = {
        "gt_f0": [np.zeros((2, 3, 3), np.uint8)],
        "gt_f1": [np.ones((2, 3, 3), np.uint8)],
        "lstars": [np.zeros((2, 3), np.int64)],
        "margins": [np.ones((2, 3), np.float32)],
        "gt_poses": [np.arange(6, dtype=np.float64)],
    }
    digest = fresh_training_target_sha256(fields)
    changed = {**fields, "margins": [np.zeros((2, 3), np.float32)]}
    assert len(digest) == 64
    assert fresh_training_target_sha256(fields) == digest
    assert fresh_training_target_sha256(changed) != digest
    with pytest.raises(ValueError, match="missing field"):
        fresh_training_target_sha256({"lstars": fields["lstars"]})


def test_resume_registry_roundtrip_preserves_derived_selection() -> None:
    source = _applied_state()
    arrays = fresh_state_arrays(source)
    assert set(arrays) == {
        FRESH_RESUME_PREFIX + suffix
        for suffix in (
            "version",
            "enabled",
            "control",
            "applied",
            "candidate_index",
            "selected_freq_along",
            "selected_bias_k",
            "selected_mean_distance",
            "post_structured_mean_distance",
            "init_seconds",
            "selection_receipt_sha256",
            "reason",
        )
    }
    cfg = {key: value.item() for key, value in arrays.items()}
    restored = FreShInitState(enabled=True)
    assert restore_fresh_state_from_cfg(restored, cfg) is True
    assert restored.result_dict() == source.result_dict()
    assert fresh_state_arrays(FreShInitState(enabled=False)) == {}
    assert restore_fresh_state_from_cfg(restored, {}) is False


def test_applied_state_refuses_partial_or_unbound_provenance() -> None:
    with pytest.raises(ValueError, match="candidate_index"):
        fresh_state_arrays(FreShInitState(enabled=True, applied=True))
    bad = _applied_state()
    bad.selection_receipt_sha256 = "not-a-sha"
    with pytest.raises(ValueError, match="SHA-256"):
        fresh_state_arrays(bad)


def test_checkpoint_cfg_restores_frequency_before_directional_features() -> None:
    source = _applied_state()
    cfg = {key: value.item() for key, value in fresh_checkpoint_cfg_arrays(source).items()}
    args = _args(freq_along=8.0)
    restored = FreShInitState(enabled=True)
    assert restore_fresh_checkpoint_before_features(args, restored, cfg) is True
    assert args.freq_along == pytest.approx(source.selected_freq_along)
    assert restored.selected_bias_k == pytest.approx(source.selected_bias_k)
    assert restored.reason == "restored_from_checkpoint"


def test_nonfresh_resume_is_explicitly_overwriting_not_reselected() -> None:
    args = _args()
    state = FreShInitState(enabled=True)
    assert restore_fresh_checkpoint_before_features(args, state, {"__cfg_fresh_init": 0}) is False
    assert state.applied is False
    assert state.reason == "overwritten_by_nonfresh_resume"
    assert args.freq_along == 8.0


def test_nonfresh_resume_two_hop_stays_explicitly_nonfresh() -> None:
    args = _args()
    first = FreShInitState(enabled=True)
    assert restore_fresh_checkpoint_before_features(args, first, {"__cfg_fresh_init": 0}) is False
    compact = fresh_checkpoint_cfg_arrays(first, args=args)
    assert int(compact["__cfg_fresh_init"]) == 0
    assert int(compact["__cfg_fresh_requested"]) == 1
    second = FreShInitState(enabled=True)
    cfg = {key: value.item() for key, value in compact.items()}
    assert restore_fresh_checkpoint_before_features(_args(), second, cfg) is False
    assert second.reason == "overwritten_by_nonfresh_resume"


def test_checkpoint_cfg_reader_ignores_weight_payload(tmp_path: Path) -> None:
    path = tmp_path / "resume.npz"
    np.savez(
        path,
        **fresh_checkpoint_cfg_arrays(_applied_state()),
        **fresh_state_arrays(_applied_state()),
        **{"__live__hidden.weight": np.ones((4, 4), np.float32)},
    )
    cfg = load_checkpoint_cfg(path)
    assert "__cfg_fresh_init" in cfg
    assert FRESH_RESUME_PREFIX + "enabled" in cfg
    assert "__live__hidden.weight" not in cfg
