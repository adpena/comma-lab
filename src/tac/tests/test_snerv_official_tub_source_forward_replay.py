# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tac.analysis.snerv_official_tub_source_forward_replay import (
    DEFAULT_OFFICIAL_SNERV_REPO,
    PYTORCH_WAVELETS_BLOCKER,
    SCHEMA,
    STATE_VALUE_ARTIFACT_BLOCKER,
    TUB_CLOSED_BY_FIXTURE_REPLAY,
    TUB_PRESERVED_BLOCKERS,
    build_snerv_official_tub_source_forward_replay_artifact,
    main,
)
from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    OFFICIAL_SNERV_T_SOURCE_SHA,
)


def _official_repo() -> Path:
    if not DEFAULT_OFFICIAL_SNERV_REPO.exists():
        pytest.skip(f"official SNeRV checkout is absent: {DEFAULT_OFFICIAL_SNERV_REPO}")
    return DEFAULT_OFFICIAL_SNERV_REPO


def test_snerv_official_tub_source_forward_replay_executes_output2_path() -> None:
    artifact = build_snerv_official_tub_source_forward_replay_artifact(
        official_repo_dir=_official_repo(),
        generated_utc="20260604T000000Z",
    )

    assert artifact["schema"] == SCHEMA
    assert artifact["official_repo"]["head_sha"] == OFFICIAL_SNERV_T_SOURCE_SHA
    assert artifact["source_forward_replay_executed"] is True
    assert (
        artifact["official_tub_temporal_encoder_output2_source_fixture_replay_passed"]
        is True
    )
    assert artifact["source_forward_parity_proven"] is False
    assert artifact["full_tub_source_forward_parity_proven"] is False
    assert artifact["official_trained_checkpoint_loaded"] is False
    assert artifact["score_claim"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False

    graph = artifact["graph_input_parity"]
    assert graph["graph_input_parity_passed"] is True
    assert graph["max_abs_error"] == 0.0
    assert graph["output_hashes_bit_identical"] is True

    fusion = artifact["portable_output2_fusion"]
    assert fusion["portable_output2_fusion_receiver_mapping_proven"] is True
    assert fusion["max_abs_error"] == 0.0
    assert fusion["output_hashes_bit_identical"] is True
    assert fusion["output_shapes"]["output2_decoder_input"] == [2, 3, 1, 1]
    assert fusion["output_shapes"]["output2_shuffled"] == [2, 6, 2, 2]
    assert (
        "snerv_official_tub_portable_output2_fusion_receiver_mapping_missing"
        in fusion["closed_blockers"]
    )
    assert len(fusion["source_output_sha256"]) == 64

    temporal = artifact["temporal_path"]
    assert temporal["source_forward_fixture_replay_passed"] is True
    assert temporal["output_tensors_finite"] is True
    assert temporal["official_module_classes"]["encoder1"] == "ConvNeXt"
    assert temporal["official_module_classes"]["output2_decoder"] == "Sequential"
    assert temporal["output_shapes"]["output2_raw"] == [2, 6, 2, 2]
    assert temporal["output_shapes"]["output2_shuffled"] == [2, 6, 2, 2]
    assert temporal["output_shapes"]["final_decoder_output"] == [1, 3, 16, 16]
    assert len(temporal["output_sha256"]) == 64

    equivalence = artifact["full_forward_equivalence"]
    assert equivalence["manual_replay_matches_official_forward"] is True
    assert equivalence["max_abs_error"] == 0.0
    assert equivalence["output_hashes_bit_identical"] is True

    for blocker in TUB_CLOSED_BY_FIXTURE_REPLAY:
        assert blocker in artifact["closed_blockers"]
    for blocker in TUB_PRESERVED_BLOCKERS:
        assert blocker in artifact["preserved_blockers"]


def test_snerv_official_tub_replay_preserves_dependency_and_checkpoint_blockers() -> None:
    artifact = build_snerv_official_tub_source_forward_replay_artifact(
        official_repo_dir=_official_repo(),
        generated_utc="20260604T000000Z",
    )

    assert artifact["dependency_contract"]["functional_haar_shim_used_for_fixture"] is True
    assert artifact["dependency_contract"]["shim_score_authority"] is False
    assert artifact["source_fixture_not_training_config"] is True
    assert "snerv_official_trained_checkpoint_state_dict_not_loaded" in artifact["blockers"]
    assert (
        "snerv_official_tub_portable_temporal_encoder_output2_receiver_mapping_missing"
        not in artifact["blockers"]
    )
    assert (
        "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing"
        in artifact["blockers"]
    )
    assert (
        "snerv_official_tub_portable_output2_decoder_weight_mapping_missing"
        in artifact["blockers"]
    )
    if importlib.util.find_spec("pytorch_wavelets") is None:
        assert PYTORCH_WAVELETS_BLOCKER in artifact["blockers"]
        assert (
            artifact["official_pytorch_wavelets_runtime_dependency_installed"] is False
        )
    else:
        assert artifact["official_pytorch_wavelets_runtime_dependency_installed"] is True


def test_snerv_official_tub_source_forward_replay_persists_value_state_npz(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "official_snerv_t_one_step_state_dict.npz"
    artifact = build_snerv_official_tub_source_forward_replay_artifact(
        official_repo_dir=_official_repo(),
        train_one_step=True,
        output_state_dict_path=state_path,
        generated_utc="20260604T000000Z",
    )

    assert state_path.is_file()
    assert artifact["source_forward_replay_executed"] is True
    assert artifact["official_trained_checkpoint_loaded"] is True
    assert artifact["official_trained_checkpoint_state_dict_mapping_verified"] is True
    assert artifact["full_tub_source_forward_parity_proven"] is True
    assert (
        artifact["official_trained_checkpoint_state_dict_value_artifact_ready"] is True
    )
    assert artifact["source_forward_replay_authority"] is True
    assert STATE_VALUE_ARTIFACT_BLOCKER not in artifact["blockers"]
    assert artifact["official_trained_checkpoint_state_dict_path"] == state_path.as_posix()
    assert artifact["official_trained_checkpoint_state_dict_slice_present"] is True
    assert artifact["official_trained_checkpoint_state_dict_slice_file_present"] is True
    assert (
        artifact["official_trained_checkpoint_state_dict_slice_runner_arg"]
        == "--snerv-official-trained-checkpoint-state-dict-path"
    )
    state_artifact = artifact["official_trained_checkpoint_state_dict_artifact"]
    assert state_artifact["path"] == state_path.as_posix()
    assert state_artifact["bytes"] == state_path.stat().st_size
    assert len(state_artifact["sha256"]) == 64
    assert state_artifact["member_count"] > 0
    assert any(
        name.startswith("encoder.1.") and name.endswith(".npy")
        for name in state_artifact["member_names"]
    )
    assert any(
        name.startswith("encoder.2.") and name.endswith(".npy")
        for name in state_artifact["member_names"]
    )
    assert any(
        name.startswith("decoder.") and name.endswith(".npy")
        for name in state_artifact["member_names"]
    )
    mapping = artifact["official_trained_checkpoint_mapping_manifest"]
    assert mapping["official_tub_temporal_encoder_weight_mapping_proven"] is True
    assert mapping["official_tub_output2_decoder_weight_mapping_proven"] is True
    assert mapping["official_mfu_trained_checkpoint_weight_mapping_proven"] is True
    with np.load(state_path, allow_pickle=False) as data:
        keys = list(data.files)
        assert any(key.startswith("encoder.1.") for key in keys)
        assert any(key.startswith("encoder.2.") for key in keys)
        assert any(key.startswith("decoder.") for key in keys)


def test_snerv_official_tub_source_forward_replay_blocks_authority_without_value_state_npz() -> None:
    artifact = build_snerv_official_tub_source_forward_replay_artifact(
        official_repo_dir=_official_repo(),
        train_one_step=True,
        generated_utc="20260604T000000Z",
    )

    assert artifact["source_forward_replay_executed"] is True
    assert artifact["official_trained_checkpoint_loaded"] is True
    assert artifact["official_trained_checkpoint_state_dict_mapping_verified"] is True
    assert artifact["full_tub_source_forward_parity_proven"] is True
    assert (
        artifact["official_trained_checkpoint_state_dict_value_artifact_ready"] is False
    )
    assert artifact["source_forward_replay_authority"] is False
    assert STATE_VALUE_ARTIFACT_BLOCKER in artifact["blockers"]


def test_snerv_official_tub_replay_missing_checkout_is_fail_closed(tmp_path: Path) -> None:
    artifact = build_snerv_official_tub_source_forward_replay_artifact(
        official_repo_dir=tmp_path / "missing-snerv",
        generated_utc="20260604T000000Z",
    )

    assert artifact["schema"] == SCHEMA
    assert artifact["source_forward_replay_executed"] is False
    assert artifact["official_tub_temporal_encoder_output2_source_fixture_replay_passed"] is False
    assert artifact["score_claim"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False
    assert "snerv_official_source_checkout_missing" in artifact["blockers"]


def test_snerv_official_tub_replay_cli_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "tub_replay.json"

    rc = main(
        [
            "--official-repo-dir",
            _official_repo().as_posix(),
            "--generated-utc",
            "20260604T000000Z",
            "--write-json",
            out.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["source_forward_replay_executed"] is True
    assert payload["score_claim"] is False


def test_snerv_official_tub_replay_cli_writes_value_state_npz(
    tmp_path: Path,
) -> None:
    out = tmp_path / "tub_replay.json"
    state_path = tmp_path / "official_state.npz"

    rc = main(
        [
            "--official-repo-dir",
            _official_repo().as_posix(),
            "--generated-utc",
            "20260604T000000Z",
            "--train-one-step",
            "--write-state-dict-npz",
            state_path.as_posix(),
            "--write-json",
            out.as_posix(),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert state_path.is_file()
    assert payload["official_trained_checkpoint_state_dict_path"] == (
        state_path.as_posix()
    )
    assert payload["official_trained_checkpoint_state_dict_slice_present"] is True
    assert payload["official_trained_checkpoint_state_dict_artifact"]["member_count"] > 0
    assert payload["official_trained_checkpoint_state_dict_value_artifact_ready"] is True
    assert payload["source_forward_replay_authority"] is True
