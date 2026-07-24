from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tools.measure_ddm_ws1_seg_lexicographic_warmstart import (
    CONFIG_SCHEMA,
    POINTER,
    WS1Config,
)
from tools.measure_ddm_ws1_seglex96_filtered_warmstart import FilteredConfig

CONFIG = (
    Path(__file__).resolve().parents[2]
    / ".omx/research/configs/ddm_ws1_seg_lexicographic_warmstart_20260724.json"
)
FILTERED_CONFIG = (
    Path(__file__).resolve().parents[2]
    / ".omx/research/configs/ddm_ws1_seglex96_filtered_warmstart_20260724.json"
)


def test_ws1_config_is_local_only_and_thread_pinned() -> None:
    config = WS1Config.model_validate_json(CONFIG.read_bytes())
    assert config.schema_ == CONFIG_SCHEMA
    assert config.scorer_threads == 4
    assert config.pointer == POINTER
    assert config.paid_dispatch_allowed is False
    assert config.exact_eval_allowed is False
    assert config.frontier_mutation_allowed is False
    assert config.training_allowed is False
    assert config.score_claim is False


def test_ws1_config_rejects_local_checkpoint_root() -> None:
    payload = json.loads(CONFIG.read_bytes())
    payload["checkpoint_root"] = "/tmp/not-durable"
    with pytest.raises(ValidationError, match="primary SSD tier"):
        WS1Config.model_validate(payload)


def test_ws1_config_rejects_erased_mycar_guard() -> None:
    payload = json.loads(CONFIG.read_bytes())
    payload["mycar_material_defect_ceiling"] = payload["mycar_control_errors"]
    with pytest.raises(ValidationError, match="40000"):
        WS1Config.model_validate(payload)


def test_filtered_ws1_config_binds_exact_seglex96_archive() -> None:
    config = FilteredConfig.model_validate_json(FILTERED_CONFIG.read_bytes())
    assert config.filtered_archive_bytes == 137827
    assert (
        config.filtered_archive_sha256
        == "4fbba057b10c64d85f73ea2da3287f5fbd3f794c71ef0762fe0e0e50a224ea2d"
    )
    assert config.training_allowed is False
    assert config.exact_eval_allowed is False
    assert config.score_claim is False
