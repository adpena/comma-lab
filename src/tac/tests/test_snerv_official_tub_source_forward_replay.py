# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tac.analysis.snerv_official_tub_source_forward_replay import (
    DEFAULT_OFFICIAL_SNERV_REPO,
    PYTORCH_WAVELETS_BLOCKER,
    SCHEMA,
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
